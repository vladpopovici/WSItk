__author__ = 'vlad'

import math
import random
import numpy as np

def gibbs(N=500,thin=10):
    x, y = 0, 0
    k = 0
    xy = np.ndarray((N,2), dtype=np.float64)

    for i in range(N):
        for j in range(thin):
            x=random.gammavariate(3,1.0/(y*y+4))
            y=random.gauss(1.0/(x+1),1.0/math.sqrt(2*x+2))
        xy[k,:] = x, y
        k += 1

    return xy
