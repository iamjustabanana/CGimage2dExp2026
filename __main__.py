import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.__main__

sys.exit(src.__main__.main())
