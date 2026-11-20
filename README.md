# comfyui-kimodo-motion：Kimodo 动作生成节点

给 demo-server 生成服务的动作应用（内置模板 `kimodo-motion`）用的 ComfyUI 节点。生成直接调上游 Kimodo（[nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo)，`install.sh` 钉住提交、装在本目录的 `vendor/` 下），不依赖 ComfyUI-Kimodo-Enhanced 的采样节点。

- `DemoKimodoGenerate`：提示词（一句）+ 约束 JSON 的文本（Kimodo `constraints.md` 的数组格式，空串 = 不加约束）→ `KIMODO_MOTION`。输入：`duration`（秒，帧数 = `int(duration × 30)`）、`seed`、`diffusion_steps`、`model`（默认 `Kimodo-SOMA-RP-v1.2`）、`cfg_text` / `cfg_constraint`（分离的 CFG，默认都是 2）、`post_processing`（默认开）、`root_margin`。
- `DemoKimodoSaveMotion`（输出节点）：`KIMODO_MOTION` → 输出目录下的 `<前缀>_<序号>_.motion.json`，返回 `ui.files`，comfy-api-proxy 据此把它列成任务产物。
- `DemoKimodoConstraints`：约束文本 → 临时文件路径，接 `Kimodo_Sampler.constraints_json` 用（旧模板留下的，新模板不用）。

**和 demo-server 的约定**（改了要两边一起改）：节点类名和输入名（demo-server 的工作流模板 `kimodo-motion` 按它们写）、`.motion.json` 的形状（生成服务按文件名后缀认成动作产物，导演台前端读它做重定向；见 `__init__.py` 文件头，字段名不要改）。导出骨架的全部关节（SOMA 是 77 个，含手指；读的一方按名字取，身体那 22 个 `BODY_JOINTS` 一定有），带 `parents`（父关节下标，导演台的关键帧补帧靠它把姿态换成约束要的本地旋转），坐标同 Kimodo：米、Y 向上、+Z 朝前，旋转是全局四元数（xyzw）。

## 生成按上游的推荐做法

出处是上游文档的 Best Practices、Constraints、Generation Parameters 三页：

- **模型**：`Kimodo-SOMA-RP-v1.2`，全量 Rigplay（700 小时）训练的最新版，上游演示和命令行的默认。SEED 版只有 288 小时数据、能力弱；ArdyRep 版把原始髋当根，没有路径约束用的平滑根。
- **后处理开着**：模型自己会脚滑、也不会精确落到约束上，后处理（`motion_correction`，C++ 扩展）清脚滑并把约束帧拉准。上游 2026-04-24 修过它的一个缺陷（整身关键帧的根被拉向平滑根），所以要用钉住的这个提交，不用别的节点包自带的旧版。
- **分离的 CFG**：文字、约束各一个权重，默认 2 / 2；约束和文字打架时调。
- **关 TF32**：它给生成的动作加逐帧的小噪声、脚滑变多；节点只在自己采样时关，不影响别的节点。
- **权重放在 `models/Kimodo/<模型名>/`**（持久卷），第一次用时下载、下完才算数，之后加载不联网。
- 文本编码器外置（见下面「环境变量」）。

## 环境变量

配在 **ComfyUI 进程**的环境里（k8s 是容器的 env / ConfigMap），不是装节点时的环境；改了要重启 ComfyUI。本节点包自己不定义环境变量，下面都是上游 Kimodo 和 huggingface_hub 读的，示例见 `.env.example`。ComfyUI 启动时节点会在日志里打一行 `[DemoKimodo] TEXT_ENCODER_MODE=… TEXT_ENCODER_URL=…`，没配对会跟一行警告。

| 变量 | 要不要配 | 说明 |
|---|---|---|
| `TEXT_ENCODER_MODE` | **必配 `api`** | 文本编码器（Llama 3 8B + LLM2Vec）走外置服务。不配是 `auto`：连不上服务就在 ComfyUI 进程里加载 Llama，Kimodo 要的 transformers 5.x 和共享环境通常不兼容。 |
| `TEXT_ENCODER_URL` | **必配** | 外置文本编码器服务的地址（上游 `kimodo/scripts/run_text_encoder_server.py` 起的 Gradio 服务），不配是 `http://127.0.0.1:9550/`。 |
| `NO_PROXY` / `no_proxy` | 配了代理时 | 要盖住 `TEXT_ENCODER_URL` 的主机名。按域名后缀写（如 `.svc.cluster.local`）：httpx 不认 `192.168.0.0/16` 这种网段写法，写 IP 的地址会被送去代理。 |
| `HTTP_PROXY` / `HTTPS_PROXY` | 看网络 | 第一次用某个模型时下载权重、`install.sh` 取上游源码和 Eigen 都要出网。 |
| `HF_ENDPOINT` | 看网络 | 权重从 Hugging Face 下（`nvidia/<模型名>`），连不上时指到镜像。 |

不用配的：`CHECKPOINT_DIR` 由节点自己设成 `models/Kimodo`（配了也会被盖掉）；`TEXT_ENCODER`、`TEXT_ENCODERS_DIR`、`TEXT_ENCODER_DEVICE`、`HUGGINGFACE_CACHE_DIR` 只在本地加载文本编码器（`TEXT_ENCODER_MODE=local`）时才读。

安装时的变量（跑 `install.sh` 时给，Manager 自动跑时用不上）：`PYTHON`（用哪个 python，`install.py` 会设成 ComfyUI 的那个）、`SKIP_LINK=1`（不切换 ComfyUI-Kimodo-Enhanced 的 `kimodo`）。

## 安装

首次（二选一）：

- 在 ComfyUI 的环境里 `cd custom_nodes && git clone https://github.com/fyl080801/comfyui-kimodo-motion && python comfyui-kimodo-motion/install.py`；
- 或 ComfyUI-Manager 的「Install via Git URL」填仓库地址（ComfyUI 不是只监听本机时，Manager 的 `security_level` 要是 `weak` 才放行）。Manager 装完自动跑 `install.py`。

`install.py` 只是用 ComfyUI 的那个 python 跑 `install.sh`：取上游 Kimodo 的那个提交到 `vendor/kimodo`，把 `motion_correction` 编进 `vendor/site`（要 cmake、g++、能连 GitHub / GitLab；只支持 x86-64，扩展用了 SSE / AVX）。不往共享环境里装任何包。同一个进程里只能有一份 `kimodo`：装了 ComfyUI-Kimodo-Enhanced 时，脚本把它自带的那份换成指向 `vendor/kimodo` 的链接（原来的留作 `kimodo.bundled`；已经是链接但指向别处时改指过来）；`SKIP_LINK=1` 只装不切换。

装完重启 ComfyUI。启动日志里没有报错、`GET /object_info` 里能看到 `DemoKimodoGenerate`、`DemoKimodoSaveMotion` 就是装好了。

依赖共享环境里已有的 torch、hydra-core、omegaconf、einops、peft、transformers、gradio_client、huggingface_hub。

## 更新

在 ComfyUI-Manager 的节点列表里对本节点包点 Update（= `git pull` 后再跑一次 `install.py`），然后重启 ComfyUI（custom node 只在启动时加载；有任务在跑时等队列空了再重启）。上游提交没变时 `install.py` 不重新编译，几秒就完。

- 跟的是默认分支的最新提交，Manager 对不在 Comfy Registry 上的节点包不能选版本；要回退就在目录里手工 `git checkout <tag 或提交>` 再跑 `install.py`。
- 换上游 Kimodo 版本：改 `install.sh` 的 `KIMODO_COMMIT` 提交，照常更新。
- `install.py` 失败（缺编译器、连不上 GitHub）时报错只在 ComfyUI 的日志里，节点用到后处理时会报「重跑 install.sh」。

## 从拷贝安装的 `demo_kimodo` 迁过来

早先是把 demo-server 仓库里的 `demo_kimodo` 目录手工拷进 `custom_nodes/`。两个目录同时在会注册同名节点，迁的时候：

1. 按上面装好本节点包（`install.sh` 会把 ComfyUI-Kimodo-Enhanced 的 `kimodo` 链接改指到新目录）；嫌重新编译慢可以先 `mv custom_nodes/demo_kimodo/vendor custom_nodes/comfyui-kimodo-motion/vendor` 再跑 `install.py`（旧目录编的没有提交标记，仍会重编一次扩展，上游源码不用重取）。
2. 删掉 `custom_nodes/demo_kimodo`。
3. 重启 ComfyUI。
