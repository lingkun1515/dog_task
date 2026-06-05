"""dog_sim 包的可执行入口.

通过 ``python3 -m dog_sim`` 运行时, Python 会执行本文件.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
