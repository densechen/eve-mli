import os
import sys
sys.path.insert(0, os.path.abspath(".."))

from pprint import pprint
print("Python library search path")
pprint(sys.path[0])