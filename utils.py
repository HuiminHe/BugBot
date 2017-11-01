import numpy as np
from scipy.spatial.distance import cdist
from simulator import Map
def dist(a, b):
    assert type(a) != type(Map)
    scale = a.env.scale
    pa = a.geom.v / scale  + a.loc()
    pb = b.geom.v / scale  + b.loc()
    return np.amin(cdist(pa, pb))

def dire(v):
    return v / np.linalg.norm(v)

