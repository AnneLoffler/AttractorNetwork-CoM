"""
Local optimization script.
"""

import numpy as np
from ANM import ANN

def f(X): # coh, col, hierarchy, W_sensory, W_sensory_auto, W_intentional, W_cost, W_inhibit):
    s1,s2,s3,s4,s5,s6,s7,s8,s9 = ANN(X, n_trials=10000)
    cost = s1+s2+s3+s4+s5+s6+s7+s8+s9
    print("COST", cost)
    return cost



# LBounds = [0 0 0 0 0 0 -2 -2]'; UBounds = [60 60 2 2 1 2 0 0]';
# all connections in network should have absolute values between 0-2
s = {"coh"            : (0.01, 60.0),
     "col"            : (0.01, 60.0),
     "hierarchy"      : (0.01, 2.0),
     "W_sensory"      : (0.01, 2.0),
     "W_sensory_auto" : (0.01, 1.0),
     "W_intentional"  : (0.01, 2.0),
     "W_cost"         : (-2.0, -0.01),
     "W_inhibit"      : (-2.0, -0.01)}

x0 = np.array([3.2, 51.0, 1.0, 1.5, 0.25, 1, -1, -0.5])

from scipy import optimize

res1 = optimize.fmin_powell(f, x0)

print("Result:", res1)
