import sys , os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print (base_dir)
print ("base_dir")
sys.path.append(base_dir)

import food.proprecess.proprecess