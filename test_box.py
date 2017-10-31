from simulator import Simulator, Map, Agent
from devices import Device
import numpy as np
import simulator_config

env = Simulator(simulator_config)
map = Map()
map.get_map_from_geom2d(env, kp=np.array([[-100, 100], [-100, -100], [100, -100], [100, 100]]), n_pts=180)

robot = Agent(env, kp=np.array([[-2, 0], [2, 0]]), color=(1, 0, 0, 0.5), v_max=5)
robot.reset(init_state=np.array([0, 40, 0]))
device = Device(env, parent=robot, kp=np.array([[-10, 0], [10, 0]]), color=[0, 1, 0, 1], filled=False)
while True:
    robot.update(v=np.array([5, 0]))
    env._render()
