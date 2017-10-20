from simulator import Simulator, Map, Agent
import numpy as np
from time import time

env = Simulator()
map = Map()
map.get_map_from_geom2d(env, radius=100, n_pts=180)

n_targets = 10
n_robots = 10

targets = [Agent(env, radius=3, color=(1, 0, 0, 0.5), v_max=1.5) for i in range(n_targets)]
robots = [Agent(env, radius=3, color=(1, 0, 1, 0.5), v_max=2) for i in range(n_targets, n_robots+n_targets)]
tic = time()
vs = (np.random.rand(n_targets+n_robots,2) - 0.5) * 4


while True:
    env._render()
    for i in range(n_targets + n_robots):
        if np.random.rand() < 0.02:
            vs[i] = (np.random.rand(2) - 0.5) * 4
        else:
            vs[i] = vs[i]
    for i, t in enumerate(targets+robots):
        t.update(v=np.array(vs[i]))
    print('update takes {} sec'.format(time() - tic))
    # print()
    tic = time()

