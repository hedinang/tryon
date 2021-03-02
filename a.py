import numpy as np
import torch
a = [[3, 2, 3], [3, 3, 2]]
a = np.array(a)
b = (a==3) .astype(np.int)
print(b)
