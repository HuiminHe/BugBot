import numpy as np
import scipy.spatial.distance

def dist(a, b):
    pa = a.state[:2]
    pb = b.state[:2]
    return np.linalg.norm(pb-pa)

def dire(v):
    return v / np.linalg.norm(v)

