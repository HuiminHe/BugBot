from simulator import Simulator, Map, Agent
from devices import Blob_finder
import numpy as np


env = Simulator()
map = Map()
map.get_map_from_geom2d(env, radius=100, n_pts=180)

robot = Agent(env, radius=3, color=(1, 0, 0, 0.5), v_max=1.5)
robot.reset(init_state=np.array([70, 40, 0]))
blob_finder = Blob_finder(env, parent=robot, range=30, color=[0, 1, 0, 0.1])
robot.add_device(blob_finder)
while True:
    env._render()
    robot.update(v=np.array([1, 0]))
    robot.devices[0]._read()