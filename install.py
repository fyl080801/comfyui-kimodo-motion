"""ComfyUI-Manager 安装、更新本节点包之后自动执行的入口：用 ComfyUI 的那个 python 跑 install.sh。失败时退出码非零，报错在 ComfyUI 的日志里。"""

import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, PYTHON=sys.executable)
sys.exit(subprocess.call(["sh", os.path.join(here, "install.sh")], env=env))
