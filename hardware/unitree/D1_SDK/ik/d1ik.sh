#!/usr/bin/env bash
# D1 逆解工具便捷入口, 自动注入本地依赖路径 ./pylibs.
# 用法等同于 solve_ik.py, 例如:
#   ./d1ik.sh solve --xyz 0.3 0 0.2 --approach-dir 0 0 -1
#   ./d1ik.sh grasp --bottle 0.3 0 0.05 --basket 0.1 0.25 0.1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="${HERE}/pylibs" python3 "${HERE}/solve_ik.py" "$@"
