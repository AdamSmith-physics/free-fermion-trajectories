# put this at the top of your files: "from fix_pathing import root_dir"

import os
import sys

root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)