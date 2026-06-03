from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PARENT_CORE = ROOT / "packages" / "parent-core"
if str(PARENT_CORE) not in sys.path:
    sys.path.insert(0, str(PARENT_CORE))

from optimaize.app_ui import main


if __name__ == "__main__":
    main()
