import sys
import os
# pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# sys.path.insert(0,pythonpath)

import sys , os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

print ("main base_dir")
print (base_dir)

import food.proprecess.proprecess

import food.examples.examples