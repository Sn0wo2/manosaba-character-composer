from pathlib import Path
import re
import io
import soundfile as sf
from collections import Counter
import logging
import os
import sys
from typing import Literal, Optional, Dict, Any, Tuple, List
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, as_completed
import json
import threading
import ctypes
import signal
from PIL import Image

from tqdm import tqdm

import UnityPy
from UnityPy.files import ObjectReader
from UnityPy.classes import TextAsset, Texture2D, AudioClip, AssetBundle, Sprite, GameObject, Transform, SpriteRenderer, EditorExtension

ILLEGAL_CHARS_RE = re.compile(r'[<>:"/\|?*#]')

def _sanitize_name(name: str) -> str:
    """替换文件名/路径中不合法的字符为下划线"""
    return ILLEGAL_CHARS_RE.sub('_', name)

# 进程池调用的全局函数，用于保存图片，避开 GIL
def _save_image_worker(img_data: bytes, out_path: str, save_format: str, save_kwargs: dict):
    try:
        img = Image.open(io.BytesIO(img_data))
        out_path_obj = Path(out_path)
        out_path_obj.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path_obj, format=save_format, **save_kwargs)
        return True
    except Exception as e:
        return str(e)

# 辅助函数：在树中查找节点
def _find_node(container: Dict[str, Any], target_id: str) -> Optional[Dict[str, Any]]:
    if target_id in container:
        return container[target_id]
    for v in container.values():
        children = v.get("Children")
        if isinstance(children, dict):
            found = _find_node(children, target_id)
            if found:
                return found
    return None

# 辅助函数：收集所有 ParentId == target_id 的节点
def _collect_children(container: Dict[str, Any], target_parent_id: str, acc: list):
    for k, v in list(container.items()):
        if v.get("ParentId") == target_parent_id:
            acc.append((container, k, v))
        else:
            ch = v.get("Children")
            if isinstance(ch, dict):
                _collect_children(ch, target_parent_id, acc)

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    if is_admin():
        return True
    else:
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        return False

class AssetBundleExtractor:
    def __init__(self, input_dir, output_dir, use_logger=False, max_workers=None, logger=None, is_debug=False, skip_exists_dir=False, skip_AssetBundle=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_logger = use_logger
        self.is_debug = is_debug
        self.skip_exists_dir = skip_exists_dir
        self.skip_AssetBundle = skip_AssetBundle
        self.handlers = {
            "TextAsset": self._handle_text_asset,
            "Texture2D": self._handle_texture,
            "AudioClip": self._handle_audioclip,
            "AssetBundle": self._handle_assetbundle,
            "Sprite": self._handle_texture,
            "GameObject": self._handle_gameobject,
        }
        self.processed_objects = set()
        
        if max_workers is None:
            max_workers = os.cpu_count() or 4
            
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.img_executor = ProcessPoolExecutor(max_workers=max_workers)
        
        self.task_semaphore = threading.Semaphore(max_workers * 50)
        
        self.type_counter = Counter()
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        self.pbar = None
        
        self._go_data_list = [] 
        self._go_data_lock = threading.Lock()
        
        self._stop_event = threading.Event()

    def _log(self, level: Literal["debug", "info", "warning", "error"], msg: str):
        if level == "debug" and not self.is_debug: return
        if self.use_logger or level in ["warning", "error"]:
            getattr(self.logger, level)(msg)

    def _prepare_output_dir(self, file_path: str) -> Path:
        file_path: Path = Path(file_path)
        try:
            relative_path = file_path.relative_to(self.input_dir)
        except ValueError:
            relative_path = Path(file_path.name)

        sanitized_parts = [_sanitize_name(part) for part in relative_path.parent.parts]
        sanitized_stem = _sanitize_name(file_path.stem)
        out_dir = self.output_dir.joinpath(*sanitized_parts, sanitized_stem)

        if self.skip_exists_dir and out_dir.exists() and any(out_dir.iterdir()):
            return None

        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _skip_if_exists(self, path: Path) -> bool:
        if path.exists():
            self.type_counter["skipped"] += 1
            return True
        return False

    def _handle_text_asset(self, obj: ObjectReader, out_dir: Path):
        try:
            data: TextAsset = obj.read()
            res_name = getattr(data, "m_Name", None) or f"unnamed_{obj.path_id}"
            out_path = (out_dir / _sanitize_name(res_name)).with_suffix(".txt")
            if self._skip_if_exists(out_path): return
            out_path.write_bytes(data.m_Script.encode("utf-8", "replace"))
            self.type_counter["text"] += 1
        except: pass

    def _handle_texture(self, obj: ObjectReader, out_dir: Path):
        try:
            data: Texture2D | Sprite = obj.read()
            res_name = getattr(data, "m_Name", None) or f"unnamed_{obj.path_id}"
            out_base = out_dir / _sanitize_name(res_name)
            
            img = data.image
            if img.width > 16383 or img.height > 16383:
                out_path = out_base.with_suffix(".png")
                fmt, kwargs = "PNG", {}
            else:
                out_path = out_base.with_suffix(".webp")
                fmt, kwargs = "WEBP", {"lossless": True}

            if not self._skip_if_exists(out_path):
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                
                future = self.img_executor.submit(_save_image_worker, img_byte_arr.getvalue(), str(out_path), fmt, kwargs)
                def _done_callback(f):
                    try:
                        res = f.result()
                        if res is True: self.type_counter["image"] += 1
                        else: self._log("debug", f"图片保存失败: {out_path} | {res}")
                    except: pass
                future.add_done_callback(_done_callback)
        except Exception as e:
            self._log("debug", f"纹理读取失败: {obj.path_id} | {e}")

    def _handle_audioclip(self, obj: ObjectReader, out_dir: Path):
        try:
            data: AudioClip = obj.read()
            res_name = getattr(data, "m_Name", None) or f"unnamed_{obj.path_id}"
            out_base = out_dir / _sanitize_name(res_name)
            if self._skip_if_exists(out_base): return
            if not hasattr(data, "samples") or data.samples is None: return
            
            if isinstance(data.samples, dict):
                out_base.mkdir(exist_ok=True)
                for filename, audio_bytes in data.samples.items():
                    (out_base / filename).with_suffix(".wav").write_bytes(audio_bytes)
            else:
                out_base.with_suffix(".wav").write_bytes(data.samples)
            self.type_counter["audio"] += 1
        except: pass

    def _handle_gameobject(self, obj: ObjectReader, out_dir: Path):
        try:
            data: GameObject = obj.read()
            components = self._get_sub_components(data, ["Transform", "SpriteRenderer"])
            transform: Transform = components.get("Transform")
            sprite_renderer: SpriteRenderer = components.get("SpriteRenderer")

            parent_id = None
            if transform:
                father_ptr = getattr(transform, "m_Father", None)
                if father_ptr and hasattr(father_ptr, "read"):
                    try:
                        father_transform = father_ptr.read()
                        father_gameobject = getattr(father_transform, "m_GameObject", None)
                        if father_gameobject: parent_id = str(father_gameobject.path_id)
                    except: pass

            go_info = {
                "json_path": str((out_dir / "GameObject.json").resolve()),
                "Id": str(obj.path_id),
                "ParentId": parent_id,
                "Name": getattr(data, "m_Name", f"unnamed_{obj.path_id}"),
                "Transform": self._get_transform_info(transform) if transform else None,
                "SpriteRenderer": self._get_sprite_renderer_info(sprite_renderer) if sprite_renderer else None,
                "IsActive": getattr(data, "m_IsActive", None),
            }
            with self._go_data_lock:
                self._go_data_list.append(go_info)
            self.type_counter["gameobject"] += 1
        except: pass

    def _get_sub_components(self, data: GameObject, types: list) -> dict:
        components = {}
        for comp in getattr(data, "m_Component", []):
            pptr = getattr(comp, "component", None)
            if pptr and hasattr(pptr, "read"):
                try:
                    c_obj = pptr.read()
                    c_type = getattr(getattr(pptr, "type", None), "name", None)
                    if c_type in types: components[c_type] = c_obj
                except: continue
        return components

    def _get_transform_info(self, t: Transform):
        return {
            "Position": {"x": t.m_LocalPosition.x, "y": t.m_LocalPosition.y, "z": t.m_LocalPosition.z} if hasattr(t, "m_LocalPosition") else None,
            "Rotation": {"x": t.m_LocalRotation.x, "y": t.m_LocalRotation.y, "z": t.m_LocalRotation.z, "w": t.m_LocalRotation.w} if hasattr(t, "m_LocalRotation") else None,
            "Scale": {"x": t.m_LocalScale.x, "y": t.m_LocalScale.y, "z": t.m_LocalScale.z} if hasattr(t, "m_LocalScale") else None,
        }

    def _get_sprite_renderer_info(self, sr: SpriteRenderer):
        try:
            sprite = sr.m_Sprite.read() if getattr(sr, "m_Sprite", None) else None
            return {
                "Sprite": {"Name": sprite.m_Name, "PixelsToUnits": sprite.m_PixelsToUnits, "Pivot": {"x": sprite.m_Pivot.x, "y": sprite.m_Pivot.y}} if sprite else None,
                "Enabled": getattr(sr, "m_Enabled", None),
                "SortingOrder": getattr(sr, "m_SortingOrder", None),
                "Color": {"r": sr.m_Color.r, "g": sr.m_Color.g, "b": sr.m_Color.b, "a": sr.m_Color.a} if hasattr(sr, "m_Color") else None,
            }
        except: return None

    def _handle_assetbundle(self, obj: ObjectReader, out_dir: Path, file_path: str):
        try:
            data: AssetBundle = obj.read()
            container = getattr(data, "m_Container", {})
            for name, pptr in (container.items() if isinstance(container, dict) else container):
                if self._stop_event.is_set(): break
                if hasattr(pptr, "asset") and pptr.asset:
                    self.process_object(pptr.asset, out_dir, file_path)
        except: pass

    def extract_all(self):
        file_list = []
        for root, _, files in os.walk(self.input_dir):
            for file in files: 
                fp = os.path.join(root, file)
                if os.path.isfile(fp): file_list.append(fp)

        self.pbar = tqdm(total=len(file_list), desc="处理进度", unit="个")

        def signal_handler(sig, frame):
            print("正在紧急停止...")
            self._stop_event.set()
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.img_executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            futures = []
            for fp in file_list:
                if self._stop_event.is_set(): break
                self.task_semaphore.acquire()
                futures.append(self.executor.submit(self.process_file, fp))
                if len(futures) > self.max_workers * 20:
                    done, futures = wait(futures, timeout=0, return_when="FIRST_COMPLETED")
                    futures = list(futures)
            wait(futures)
        except KeyboardInterrupt: signal_handler(None, None)

        if self.pbar: self.pbar.close()
        self._finalize_gameobjects()
        self.img_executor.shutdown(wait=True)
        return self.type_counter

    def _finalize_gameobjects(self):
        if not self._go_data_list: return
        print("正在合并并保存 GameObject.json...")
        groups = {}
        for info in self._go_data_list:
            groups.setdefault(info["json_path"], []).append(info)
        
        for json_path, infos in groups.items():
            path_obj = Path(json_path)
            tree = {}
            if path_obj.exists():
                try:
                    with open(path_obj, "r", encoding="utf-8") as f: tree = json.load(f)
                except: pass
            
            for info in infos:
                key = info["Id"]
                if _find_node(tree, key): continue
                node = {
                    "Name": info["Name"], "Id": key, "ParentId": info["ParentId"],
                    "Transform": info["Transform"], "SpriteRenderer": info["SpriteRenderer"],
                    "IsActive": info["IsActive"], "Children": {}
                }
                if info["ParentId"]:
                    parent = _find_node(tree, info["ParentId"])
                    if parent: parent.setdefault("Children", {})[key] = node
                    else: tree[key] = node
                else: tree[key] = node
                to_move = []
                _collect_children(tree, key, to_move)
                for src_cont, src_key, src_node in to_move:
                    if src_key in src_cont: del src_cont[src_key]
                    node["Children"][src_key] = src_node

            def recursive_sort(d):
                sorted_d = dict(sorted(d.items(), key=lambda x: x[1].get("Name", "")))
                for v in sorted_d.values():
                    if "Children" in v: v["Children"] = recursive_sort(v["Children"])
                return sorted_d
            
            tree = recursive_sort(tree)
            with open(path_obj, "w", encoding="utf-8") as f:
                json.dump(tree, f, ensure_ascii=False, indent=2)

    def process_file(self, file_path: str):
        try:
            if self._stop_event.is_set(): return
            out_dir = self._prepare_output_dir(file_path)
            if out_dir is None:
                self.type_counter["skipped"] += 1
                self._update_pbar(1)
                return
            
            # 增加容错加载
            try:
                env = UnityPy.load(str(file_path))
            except Exception as e:
                self._log("debug", f"跳过无法解析的文件: {file_path} | {e}")
                self._update_pbar(1)
                return

            self._update_pbar_total(len(env.objects) - 1)
            for obj in env.objects:
                if self._stop_event.is_set(): break
                if self.skip_AssetBundle and obj.type.name == "AssetBundle": continue
                self.process_object(obj, out_dir, file_path)
                self._update_pbar(1)
            env = None
        except Exception as e:
            self._log("debug", f"处理失败: {file_path} | {e}")
            self._update_pbar(1)
        finally:
            self.task_semaphore.release()

    def _update_pbar_total(self, n):
        if self.pbar:
            with threading.Lock():
                self.pbar.total += n
                self.pbar.refresh()

    def _update_pbar(self, n=1):
        if self.pbar: self.pbar.update(n)

    def process_object(self, obj: ObjectReader, out_dir: Path, file_path: str):
        key = (file_path, obj.path_id)
        if key in self.processed_objects: return
        self.processed_objects.add(key)
        handler = self.handlers.get(obj.type.name)
        if handler:
            try:
                if obj.type.name == "AssetBundle": handler(obj, out_dir, file_path)
                else: handler(obj, out_dir)
            except: pass

if __name__ == "__main__":
    input_dir = r"D:\Steam\steamapps\common\manosaba_game\manosaba_Data\StreamingAssets\aa\StandaloneWindows64"
    output_dir = r"D:\manosaba"
    
    workers = os.cpu_count()
    
    extractor = AssetBundleExtractor(input_dir, output_dir, use_logger=True, max_workers=workers)
    print(extractor.extract_all())
