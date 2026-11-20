"""
给 demo-server 的动作生成用的 Kimodo 节点（ComfyUI custom node，由 ComfyUI-Manager 安装和更新，见 README）。生成直接调上游 Kimodo（nv-tlabs/kimodo，`install.sh` 钉住提交、装在本目录的
`vendor/` 下），不经 ComfyUI-Kimodo-Enhanced 的采样节点：那个节点包自带的 Kimodo 源码停在 v1，不开后处理、不给 CFG。

- DemoKimodoGenerate：提示词 + 约束 JSON 的文本 → KIMODO_MOTION。按上游的推荐做法：默认最新的 Rigplay 模型、分离的 CFG（文字 / 约束各一个权重）、
  开后处理（清脚滑、让结果精确落到约束上，要 `vendor/site` 里编好的 motion_correction）、采样时关 TF32；
- DemoKimodoSaveMotion：KIMODO_MOTION → 一个 `.motion.json`（骨架全部关节的全局旋转 / 位置，SOMA 是 77 个，含手指），作为产物出现在 ui.files 里，
  comfy-api-proxy 据此把它列成任务产物（Kimodo_SaveNPZ 不返回 ui，经代理拿不到）；
- DemoKimodoConstraints：约束 JSON 的文本 → 临时文件路径（接 Kimodo_Sampler.constraints_json 用，旧模板留下的，新模板不用）。

`.motion.json` 的形状是 demo-server 的跨阶段契约（生成服务转存、导演台前端重定向都按它读），字段名不要改：

    { "format": "demo-motion", "version": 1, "skeleton": "somaskel77", "fps": 30, "frames": 150,
      "joints": [...], "parents": [-1, 0, ...], "neutral": [[x,y,z], ...], "rotations": [[[x,y,z,w], ...], ...],
      "positions": [[[x,y,z], ...], ...], "contacts": [[0|1, ...], ...] }

坐标系同 Kimodo：米，Y 向上，+Z 朝前；rotations 是全局旋转四元数（xyzw）。
`joints` 是骨架的全部关节（顺序同骨架），读的一方按名字取：身体那 22 个（BODY_JOINTS）一定有，其余（手指、脚尖、头顶、眼睛、下巴）
读得懂就用、读不懂就不管。`parents` 是每个关节的父关节下标（根是 -1，父在子前），和 `joints` / `neutral` 一起就是完整的骨架：
导演台的关键帧补帧按它把关键姿态换成约束要的本地旋转。早先导出的文件只有身体 22 个关节、没有 `parents`，都是 version 1。
"""

import json
import os
import sys
import tempfile
import threading
from types import SimpleNamespace

import numpy as np

import folder_paths

FORMAT = "demo-motion"
VERSION = 1
SKELETON = "somaskel77"

# 身体关节（SOMA 名）：读的一方必须有的那些，缺了就不是能用的骨架
BODY_JOINTS = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)

LOG = "[DemoKimodo]"


class DemoKimodoConstraints:
    """约束 JSON 的文本写成临时文件，输出路径；空串输出空串（Kimodo_Sampler 见到空串就不加约束）。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "constraints": ("STRING", {"default": "", "multiline": True,
                                           "tooltip": "Kimodo 约束 JSON（数组）的文本；空 = 没有约束"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("constraints_json",)
    FUNCTION = "write"
    CATEGORY = "Kimodo/demo"

    def write(self, constraints=""):
        text = (constraints or "").strip()
        if not text:
            return ("",)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as err:
            raise ValueError(f"{LOG} 约束不是合法的 JSON：{err}") from err
        if not isinstance(data, list) or not all(isinstance(item, dict) and "type" in item for item in data):
            raise ValueError(f"{LOG} 约束必须是 JSON 数组，每项是带 type 的对象（见 Kimodo 的 constraints.md）")
        # ComfyUI 的临时目录启动时清空；文件名唯一，并发的任务互不覆盖
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".constraints.json", prefix="demo-kimodo-",
                                         dir=temp_dir, delete=False) as file:
            json.dump(data, file)
            path = file.name
        print(f"{LOG} 约束 {len(data)} 条，写到 {path}", flush=True)
        return (path,)


# —— 生成 ——

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
# 上游仓库的检出（里面是 `kimodo` 包）和编好的 motion_correction（后处理的 C++ 扩展），都由 install.sh 放好
KIMODO_SRC = os.path.join(VENDOR_DIR, "kimodo")
SITE_DIR = os.path.join(VENDOR_DIR, "site")

DEFAULT_MODEL = "Kimodo-SOMA-RP-v1.2"
# 上游的默认：分离的 CFG，文字、约束各 2.0
DEFAULT_CFG = 2.0
MIN_DURATION = 0.5
# Kimodo 一段提示词最长 10 秒
MAX_DURATION = 10.0
# 采样加后处理要的显存很少；低于这个数先让 ComfyUI 把别的模型卸掉
MIN_FREE_VRAM = 3 * 1024**3

_models = {}
_lock = threading.Lock()


def _kimodo():
    """导入上游 Kimodo（本目录 vendor 下的那一份）。进程里已经有另一份 `kimodo`（别的节点包自带的旧版）时直接报错，不带着旧版跑。"""
    if not os.path.isdir(os.path.join(KIMODO_SRC, "kimodo")):
        raise RuntimeError(f"{LOG} 没有装上游 Kimodo：在 ComfyUI 的环境里跑一次 {os.path.dirname(VENDOR_DIR)}/install.sh")
    for path in (SITE_DIR, KIMODO_SRC):
        if path not in sys.path:
            sys.path.insert(0, path)
    import kimodo

    loaded = os.path.realpath(os.path.dirname(os.path.dirname(kimodo.__file__)))
    if loaded != os.path.realpath(KIMODO_SRC):
        raise RuntimeError(
            f"{LOG} 进程里已经加载了另一份 kimodo（{loaded}），不是 {KIMODO_SRC}：重跑 install.sh"
            "（它会把 ComfyUI-Kimodo-Enhanced 自带的那份换成指向这里的链接）再重启 ComfyUI"
        )
    return kimodo


# 一个模型的检出要有的文件：缺一个就当没下完
CHECKPOINT_FILES = ("config.yaml", "model.safetensors")


def _checkpoints(name):
    """模型权重放在 ComfyUI 的 models/Kimodo/<模型名>/（持久卷）。没下完就（接着）下，下完之后加载不再联网：模型仓库连不上时照样能用。"""
    from kimodo.model.registry import KIMODO_REPO_IDS

    repo = f"nvidia/{name}"
    if repo not in KIMODO_REPO_IDS:
        known = ", ".join(r.split("/", 1)[1] for r in KIMODO_REPO_IDS)
        raise ValueError(f"{LOG} 没有模型 {name}（这一版 Kimodo 认识的：{known}）")
    root = os.path.join(folder_paths.models_dir, "Kimodo")
    path = os.path.join(root, name)

    def missing():
        return [f for f in CHECKPOINT_FILES if not os.path.isfile(os.path.join(path, f))]

    if missing():
        from huggingface_hub import snapshot_download

        print(f"{LOG} 下载模型 {repo} → {path}", flush=True)
        snapshot_download(repo_id=repo, local_dir=path)
        if missing():
            raise RuntimeError(f"{LOG} 模型 {repo} 没下全（缺 {', '.join(missing())}）：{path}")
    return root


def _load_model(name):
    """按名字加载模型（进程内缓存）。"""
    with _lock:
        if name in _models:
            return _models[name]
        kimodo = _kimodo()
        import comfy.model_management as mm

        os.environ["CHECKPOINT_DIR"] = _checkpoints(name)
        device = mm.get_torch_device()
        print(f"{LOG} 加载模型 {name}（{device}）", flush=True)
        model = kimodo.load_model(name, device=str(device))
        _models[name] = model
        return model


def _parse_constraints(text):
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise ValueError(f"{LOG} 约束不是合法的 JSON：{err}") from err
    if not isinstance(data, list) or not all(isinstance(item, dict) and "type" in item for item in data):
        raise ValueError(f"{LOG} 约束必须是 JSON 数组，每项是带 type 的对象（见 Kimodo 的 constraints.md）")
    return data


def _check_constraints(data, frames):
    """帧号要落在这一段里：越界的约束 Kimodo 不报错、只是悄悄不生效（后处理里直接丢掉），这里先拦住。"""
    for index, item in enumerate(data):
        indices = item.get("frame_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"{LOG} 第 {index + 1} 条约束（{item['type']}）没有 frame_indices")
        bad = [f for f in indices if not isinstance(f, int) or f < 0 or f >= frames]
        if bad:
            raise ValueError(
                f"{LOG} 第 {index + 1} 条约束（{item['type']}）的帧号 {bad[:5]} 不在 0 到 {frames - 1} 之间（这一段 {frames} 帧）"
            )


class DemoKimodoGenerate:
    """提示词（一句）+ 约束 → 一段动作。一次一个样本。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A person walks forward", "multiline": True,
                                      "tooltip": "一句英文描述，以 A person 开头"}),
                "duration": ("FLOAT", {"default": 5.0, "min": MIN_DURATION, "max": MAX_DURATION, "step": 0.01,
                                       "tooltip": "秒；帧数 = int(duration × 模型帧率)"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
                "diffusion_steps": ("INT", {"default": 100, "min": 10, "max": 500}),
            },
            "optional": {
                "constraints": ("STRING", {"default": "", "multiline": True,
                                           "tooltip": "Kimodo 约束 JSON（数组）的文本；空 = 没有约束"}),
                "model": ("STRING", {"default": DEFAULT_MODEL, "tooltip": "Kimodo 模型名（SOMA 骨架的）"}),
                "cfg_text": ("FLOAT", {"default": DEFAULT_CFG, "min": 0.0, "max": 10.0, "step": 0.1,
                                       "tooltip": "文字的引导权重（分离的 CFG）"}),
                "cfg_constraint": ("FLOAT", {"default": DEFAULT_CFG, "min": 0.0, "max": 10.0, "step": 0.1,
                                             "tooltip": "约束的引导权重（分离的 CFG）"}),
                "post_processing": ("BOOLEAN", {"default": True,
                                                "tooltip": "清脚滑，并让结果精确落到约束上"}),
                "root_margin": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "后处理：根离约束超过这么远（米）就修正"}),
            },
        }

    RETURN_TYPES = ("KIMODO_MOTION",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "generate"
    CATEGORY = "Kimodo/demo"

    def generate(self, prompt, duration, seed, diffusion_steps, constraints="", model=DEFAULT_MODEL,
                 cfg_text=DEFAULT_CFG, cfg_constraint=DEFAULT_CFG, post_processing=True, root_margin=0.04):
        import torch
        import comfy.model_management as mm

        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError(f"{LOG} prompt 是空的")
        name = (model or "").strip() or DEFAULT_MODEL
        data = _parse_constraints(constraints)

        device = mm.get_torch_device()
        if getattr(device, "type", "") == "cuda" and mm.get_free_memory(device) < MIN_FREE_VRAM:
            mm.unload_all_models()
            mm.soft_empty_cache()
        net = _load_model(name)
        if "soma" not in str(net.skeleton.name).lower():
            raise ValueError(f"{LOG} 只支持 SOMA 骨架的模型（{name} 是 {net.skeleton.name}）")
        kimodo = _kimodo()
        from kimodo.constraints import load_constraints_lst
        from kimodo.tools import seed_everything

        frames = int(float(duration) * net.fps)
        _check_constraints(data, frames)
        # 约束先建好：姿态换算出错（关节数不对之类）在采样之前就报出来
        constraint_lst = load_constraints_lst(data, net.skeleton) if data else []
        if post_processing:
            try:
                import motion_correction  # noqa: F401
            except ImportError as err:
                raise RuntimeError(f"{LOG} 后处理要的 motion_correction 没装好（{err}）：重跑 install.sh") from err

        print(f"{LOG} 生成 {frames} 帧，{len(data)} 条约束，{diffusion_steps} 步，"
              f"cfg 文字 {cfg_text} / 约束 {cfg_constraint}，后处理 {'开' if post_processing else '关'}", flush=True)
        # TF32 会给生成的动作加上逐帧的小噪声、脚滑变多（上游的基准测试也关掉它）；只在这次采样里关，不动 ComfyUI 别的节点
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with _lock:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            try:
                seed_everything(int(seed))
                with torch.no_grad():
                    output = net(
                        [prompt],
                        [frames],
                        num_denoising_steps=int(diffusion_steps),
                        constraint_lst=constraint_lst,
                        cfg_type="separated",
                        cfg_weight=[float(cfg_text), float(cfg_constraint)],
                        post_processing=bool(post_processing),
                        root_margin=float(root_margin),
                        return_numpy=True,
                    )
            finally:
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32

        skeleton = net.output_skeleton
        motion = SimpleNamespace(
            output_dict=output,
            model_name=name,
            skeleton_name=str(skeleton.name),
            fps=net.fps,
            texts=[prompt],
            num_frames=[frames],
            num_samples=1,
            batch_size=int(np.asarray(output["posed_joints"]).shape[0]),
            joint_parents=skeleton.joint_parents.cpu().tolist(),
            joint_names=list(skeleton.bone_order_names),
            neutral_joints=skeleton.neutral_joints.cpu().numpy(),
            skeleton=net.skeleton,
            constraint_lst=constraint_lst,
        )
        return (motion,)


def _to_numpy(value, name):
    """张量或数组 → numpy（Kimodo_Sampler 用 return_numpy=True，一般已经是 numpy；PostProcess 之后也可能是）。"""
    if value is None:
        raise ValueError(f"{LOG} 动作数据里没有 {name}")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _sample(value, name, index, batch):
    """去掉 batch 维：第 0 维等于样本数时取第 index 个（同 Kimodo_SaveNPZ 的写法）。"""
    array = _to_numpy(value, name)
    if array.ndim > 0 and array.shape[0] == batch:
        return array[index]
    return array


def _matrices_to_quaternions(mats):
    """[..., 3, 3] 旋转矩阵 → [..., 4] 四元数（xyzw，单位长度）。Shepperd 法，按对角线最大的分支算，数值稳定。"""
    m = np.asarray(mats, dtype=np.float64)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    trace = m00 + m11 + m22
    candidates = np.stack([trace, m00, m11, m22], axis=1)
    branch = np.argmax(candidates, axis=1)
    q = np.zeros((m.shape[0], 4), dtype=np.float64)

    # w 最大
    i = branch == 0
    s = np.sqrt(np.maximum(1.0 + trace[i], 1e-12)) * 2.0
    q[i, 3] = 0.25 * s
    q[i, 0] = (m[i, 2, 1] - m[i, 1, 2]) / s
    q[i, 1] = (m[i, 0, 2] - m[i, 2, 0]) / s
    q[i, 2] = (m[i, 1, 0] - m[i, 0, 1]) / s
    # x 最大
    i = branch == 1
    s = np.sqrt(np.maximum(1.0 + m00[i] - m11[i] - m22[i], 1e-12)) * 2.0
    q[i, 3] = (m[i, 2, 1] - m[i, 1, 2]) / s
    q[i, 0] = 0.25 * s
    q[i, 1] = (m[i, 0, 1] + m[i, 1, 0]) / s
    q[i, 2] = (m[i, 0, 2] + m[i, 2, 0]) / s
    # y 最大
    i = branch == 2
    s = np.sqrt(np.maximum(1.0 + m11[i] - m00[i] - m22[i], 1e-12)) * 2.0
    q[i, 3] = (m[i, 0, 2] - m[i, 2, 0]) / s
    q[i, 0] = (m[i, 0, 1] + m[i, 1, 0]) / s
    q[i, 1] = 0.25 * s
    q[i, 2] = (m[i, 1, 2] + m[i, 2, 1]) / s
    # z 最大
    i = branch == 3
    s = np.sqrt(np.maximum(1.0 + m22[i] - m00[i] - m11[i], 1e-12)) * 2.0
    q[i, 3] = (m[i, 1, 0] - m[i, 0, 1]) / s
    q[i, 0] = (m[i, 0, 2] + m[i, 2, 0]) / s
    q[i, 1] = (m[i, 1, 2] + m[i, 2, 1]) / s
    q[i, 2] = 0.25 * s

    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(*shape, 4)


def _neutral_joints(motion):
    """骨架静止姿态的关节位置 [J, 3]：先用 motion 带的，没有再问骨架对象（SOMA30 骨架的 somaskel77）。"""
    neutral = getattr(motion, "neutral_joints", None)
    if neutral is None:
        skeleton = getattr(motion, "skeleton", None)
        skeleton = getattr(skeleton, "somaskel77", skeleton)
        neutral = getattr(skeleton, "neutral_joints", None)
    if neutral is None:
        raise ValueError(f"{LOG} 动作数据里没有骨架的静止姿态（neutral_joints）")
    return _to_numpy(neutral, "neutral_joints")


def motion_document(motion, sample_index=0):
    """KIMODO_MOTION → `.motion.json` 的内容（dict）。不碰文件，单独拿出来方便排查。"""
    skeleton_name = str(getattr(motion, "skeleton_name", ""))
    if "somaskel" not in skeleton_name:
        raise ValueError(f"{LOG} 只支持 SOMA 骨架的模型（这次是 {skeleton_name or '未知骨架'}）")
    output = motion.output_dict
    batch = int(motion.batch_size)
    index = min(max(int(sample_index), 0), batch - 1)

    names = list(motion.joint_names or [])
    rotations = _sample(output.get("global_rot_mats"), "global_rot_mats", index, batch)  # [T, J, 3, 3]
    positions = _sample(output.get("posed_joints"), "posed_joints", index, batch)  # [T, J, 3]
    contacts = _sample(output.get("foot_contacts"), "foot_contacts", index, batch)  # [T, C]
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"{LOG} global_rot_mats 的形状不对：{rotations.shape}（要 [帧, 关节, 3, 3]）")
    frames, joint_count = rotations.shape[:2]
    if positions.shape != (frames, joint_count, 3):
        raise ValueError(f"{LOG} posed_joints 的形状 {positions.shape} 和 global_rot_mats 对不上")
    if len(names) != joint_count:
        raise ValueError(f"{LOG} 骨架有 {len(names)} 个关节名，动作数据却是 {joint_count} 个关节（要 somaskel77 的 77 个）")
    missing = [name for name in BODY_JOINTS if name not in names]
    if missing:
        raise ValueError(f"{LOG} 骨架里没有这些身体关节：{', '.join(missing)}")

    neutral = _neutral_joints(motion)
    if neutral.shape != (joint_count, 3):
        raise ValueError(f"{LOG} neutral_joints 的形状 {neutral.shape} 和关节数 {joint_count} 对不上")

    parents = [int(p) for p in (getattr(motion, "joint_parents", None) or [])]
    if len(parents) != joint_count or parents[0] != -1 or any(not 0 <= p < i for i, p in enumerate(parents) if i > 0):
        raise ValueError(f"{LOG} 骨架的 joint_parents 不对（要 {joint_count} 个、根是 -1、父在子前）")

    quats = _matrices_to_quaternions(rotations)
    return {
        "format": FORMAT,
        "version": VERSION,
        "skeleton": SKELETON,
        "fps": int(round(float(motion.fps))),
        "frames": int(frames),
        "joints": [str(name) for name in names],
        "parents": parents,
        "neutral": np.round(neutral.astype(np.float64), 4).tolist(),
        "rotations": np.round(quats, 5).tolist(),
        "positions": np.round(positions.astype(np.float64), 4).tolist(),
        # 触地标记可能是 bool、0/1 或概率：统一成 0/1
        "contacts": (np.asarray(contacts, dtype=np.float64) > 0.5).astype(int).reshape(frames, -1).tolist(),
    }


class DemoKimodoSaveMotion:
    """KIMODO_MOTION 存成 `<输出目录>/<子目录>/<前缀>_<序号>_.motion.json`，作为产物出现在 ui.files 里。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion": ("KIMODO_MOTION",),
                "filename_prefix": ("STRING", {"default": "demo-motion/kimodo",
                                               "tooltip": "可以带子目录（相对 ComfyUI 输出目录）"}),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 15,
                                         "tooltip": "导出第几个样本（从 0 起）"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "Kimodo/demo"
    OUTPUT_NODE = True

    def save(self, motion, filename_prefix="demo-motion/kimodo", sample_index=0):
        document = motion_document(motion, sample_index)
        output_dir = folder_paths.get_output_directory()
        # 和 SaveImage 同一套：前缀里的子目录不能跳出输出目录，序号接着已有的文件往后排
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)
        os.makedirs(full_folder, exist_ok=True)
        file = f"{filename}_{counter:05}_.motion.json"
        path = os.path.join(full_folder, file)
        with open(path, "w", encoding="utf-8") as out:
            json.dump(document, out, separators=(",", ":"))
        print(f"{LOG} 动作 {document['frames']} 帧 → {path}", flush=True)
        return {"ui": {"files": [{"filename": file, "subfolder": subfolder, "type": "output"}]}, "result": ()}


# —— 环境变量 ——


def _report_env():
    """本节点包自己不定义环境变量，要配的是上游 Kimodo 加载模型时读的那两个（README「环境变量」、`.env.example`）。
    ComfyUI 启动加载节点时打一行，没配对一眼能看出来：不挡加载，真正的报错在第一次生成时。"""
    mode = (os.environ.get("TEXT_ENCODER_MODE") or "auto").lower()
    url = os.environ.get("TEXT_ENCODER_URL") or ""
    print(f"{LOG} TEXT_ENCODER_MODE={mode} TEXT_ENCODER_URL={url or '（没配，上游默认 http://127.0.0.1:9550/）'}", flush=True)
    if mode != "api":
        print(f"{LOG} 警告：TEXT_ENCODER_MODE 不是 api，Kimodo 会在本进程里加载 Llama 文本编码器（要 transformers 5.x，和共享环境多半不兼容）", flush=True)
    elif not url:
        print(f"{LOG} 警告：TEXT_ENCODER_MODE=api 但没配 TEXT_ENCODER_URL，会去连本机的 9550", flush=True)


_report_env()

NODE_CLASS_MAPPINGS = {
    "DemoKimodoGenerate": DemoKimodoGenerate,
    "DemoKimodoConstraints": DemoKimodoConstraints,
    "DemoKimodoSaveMotion": DemoKimodoSaveMotion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DemoKimodoGenerate": "Demo Kimodo Generate (text + constraints → motion)",
    "DemoKimodoConstraints": "Demo Kimodo Constraints (text → file)",
    "DemoKimodoSaveMotion": "Demo Kimodo Save Motion (JSON)",
}
