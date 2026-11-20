#!/bin/sh
# 在 ComfyUI 的环境里（Pod 里、ComfyUI 用的那个 python）跑：把上游 Kimodo 钉在一个提交上装到本目录的 vendor/ 下，
# 并编译后处理要的 motion_correction（C++ 扩展，要 cmake、g++、能连 GitHub / GitLab 取 Eigen）。
# ComfyUI-Manager 安装、更新本节点包时经 install.py 自动跑它；可以重复跑，提交没变且扩展能导入时不重新编译。换提交改 KIMODO_COMMIT。
# 装完重启 ComfyUI。
set -eu

KIMODO_REPO=https://github.com/nv-tlabs/kimodo
# 2026-10-15：Kimodo-SOMA-RP-v1.2 成为默认模型的那次发布
KIMODO_COMMIT=e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954

HERE=$(cd "$(dirname "$0")" && pwd)
VENDOR=$HERE/vendor
SRC=$VENDOR/kimodo
SITE=$VENDOR/site
STAMP=$SITE/.kimodo-commit
PYTHON=${PYTHON:-python3}

mkdir -p "$VENDOR"
if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git init -q "$SRC"
  git -C "$SRC" remote add origin "$KIMODO_REPO"
fi
if [ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null || true)" != "$KIMODO_COMMIT" ]; then
  git -C "$SRC" fetch -q --depth 1 origin "$KIMODO_COMMIT"
  git -C "$SRC" checkout -q --detach FETCH_HEAD
fi
echo "kimodo @ $(git -C "$SRC" rev-parse HEAD)"

# 后处理的扩展：只装这一个包到 vendor/site，不动共享环境里的任何包。已经按这个提交编过、还能导入就不重编
if [ "$(cat "$STAMP" 2>/dev/null || true)" = "$KIMODO_COMMIT" ] &&
  PYTHONPATH=$SITE "$PYTHON" -c "import motion_correction" 2>/dev/null; then
  echo "motion_correction 已是这个提交的，不重编"
else
  # 编到旁边、能导入了再换上：编译要现取 pybind11 / Eigen，网络一抖就失败，失败时原来能用的那份要留着
  rm -rf "$SITE.new"
  "$PYTHON" -m pip install --quiet --no-deps --no-build-isolation --target "$SITE.new" "$SRC/MotionCorrection"
  PYTHONPATH=$SITE.new "$PYTHON" -c "import motion_correction; print('motion_correction ok')"
  echo "$KIMODO_COMMIT" >"$SITE.new/.kimodo-commit"
  rm -rf "$SITE"
  mv "$SITE.new" "$SITE"
fi

# 进程里只能有一份 kimodo（SKIP_LINK=1 跳过这一步，只装不切换）：ComfyUI-Kimodo-Enhanced 自带一份旧的（靠 sys.path 导入），换成指向这里的链接，原来的留作 kimodo.bundled
ENHANCED=$(dirname "$HERE")/ComfyUI-Kimodo-Enhanced
if [ "${SKIP_LINK:-}" != 1 ] && [ -d "$ENHANCED" ]; then
  if [ -L "$ENHANCED/kimodo" ]; then
    # 已经是链接：指向别处（本节点包换过目录）就改指到这里
    if [ "$(readlink "$ENHANCED/kimodo")" != "$SRC" ]; then
      ln -sfn "$SRC" "$ENHANCED/kimodo"
      echo "ComfyUI-Kimodo-Enhanced/kimodo -> $SRC（改了指向）"
    fi
  else
    if [ -e "$ENHANCED/kimodo.bundled" ]; then
      echo "$ENHANCED/kimodo.bundled 已经在了，没有动 $ENHANCED/kimodo：手工确认后再跑" >&2
      exit 1
    fi
    mv "$ENHANCED/kimodo" "$ENHANCED/kimodo.bundled"
    ln -s "$SRC" "$ENHANCED/kimodo"
    echo "ComfyUI-Kimodo-Enhanced/kimodo -> $SRC（原来的在 kimodo.bundled）"
  fi
fi

PYTHONPATH=$SITE:$SRC "$PYTHON" -c "import kimodo; from kimodo.model.registry import KIMODO_REPO_IDS; print('models:', ', '.join(r.split('/')[1] for r in KIMODO_REPO_IDS))"
echo "装好了，重启 ComfyUI 生效"
