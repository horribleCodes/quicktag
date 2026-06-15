"""Module entry point: python -m quicktag"""

import multiprocessing

from quicktag.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
