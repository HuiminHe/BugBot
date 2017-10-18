import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
import random
from skimage.filters import threshold_otsu

from rendering import Geom2d, Viewer, Transform
from devices import Blob_finder

collision_info = False
rebounce = 1.0

class Simulator(object):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10,
        'world_width': 200,
        'world_height': 200,
        'screen_width': 600,
        'screen_height': 600,
        'dt': 1.0 / 10,
        'eps': 0.5
    }

    def __init__(self):
        self.viewer = None
        self.state = None
        self.map = None
        self.agents = []
        self.dt = self.metadata['dt']

        self.scale = self.metadata['screen_width'] / self.metadata['world_width']
        self.move_to_center = Transform(translation=(self.metadata['screen_width'] // 2, self.metadata['screen_height'] // 2))


    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = Viewer(self.metadata['screen_width'], self.metadata['screen_height'])
            if self.map:
                geom = self.map._render()
                geom.add_attr(self.move_to_center)
                self.viewer.add_geom(geom)

            if len(self.agents):
                # randomly init the robots
                # cell_sz = int(np.amax([a.sz for a in self.agents]))
                # x = (np.array(random.sample(range(self.metadata['world_width'] // cell_sz), len(self.agents))) -
                #      self.metadata['world_width'] //cell_sz / 2) * cell_sz * 0.7
                # y = (np.array(random.sample(range(self.metadata['world_height'] // cell_sz), len(self.agents))) -
                #      self.metadata['world_height'] //cell_sz / 2) * cell_sz * 0.7
                for i, agent in enumerate(self.agents):
                    # agent.reset(init_state=np.array([x[i],y[i], 0.0]))

                    geom = agent._render()
                    geom.add_attr(self.move_to_center)
                    self.viewer.add_geom(geom)
                self.collision = CollisionDetector(self)

        if len(self.agents):
            # update agents motion
            self.collision.detect()
            for agent in self.agents:
                agent._update_render()
                for d in agent.devices:
                    geom = d._render()
                    geom.add_attr(self.move_to_center)
                    self.viewer.add_onetime(geom)


        # update collision detector
        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class Agent(Geom2d):
    counter = 0

    def __init__(self, env, radius, geom_type='circle',device=None, color=(1,0,0,0.5), v_max=2.0):
        kp = np.array([[-radius, 0], [radius, 0]])
        super().__init__(env, kp=kp, geom_type='circle', color=color, n_pts=10)
        env.agents.append(self)

        self.indx = type(self).counter
        type(self).counter += 1
        self.geom = super()._render()
        self.devices = []
        self.color = color
        self.rot = Transform()
        self.mov = Transform()
        self.v_max = v_max
        self.geom.add_attr(self.rot)
        self.geom.add_attr(self.mov)

    def _render(self) :
            return self.geom

    def _update_render(self):
        v = self.v
        x, y, a = self.state
        x += self.v[0] * self.env.dt
        y += self.v[1] * self.env.dt
        a += self.va * self.env.dt
        self.state = np.array([x, y, a])

        self.rot.set_rotation(a * self.env.dt)
        self.mov.set_translation(x * self.env.scale, y * self.env.scale)

    def reset(self, init_state=()):
        self.init_state = init_state
        if len(init_state):
            self.state = init_state
        else:
            self.state =  (np.random.rand(3) - 0.5) * self.env.metadata['world_height'] * 0.7
        self.v = rot_mat(np.array([self.v_max, 0.0]), np.random.rand()*np.pi*2)
        self.va = 0

    def update(self, v=np.array([0.0, 0.0]), va=0):
        x, y, a = self.state
        v = clip(v, max_norm=self.v_max)
        self.v = v

    def add_device(self, device):
        self.devices.append(device)


class Recorder(object):
    def __init__(self, agents):
        self.agents = agents
        self.track = []
        self.colors = []

    def _render(self) :
        for a in self.agents:
            x, y, a = a.state
            c = a.color
            self.track.append([x, y])
            self.colors.append(c)
        return


class CollisionDetector(object):
    def __init__(self, env):
        self.env = env
        self.eps = env.metadata['eps']
        self.agents = env.agents
        self.cell_sz = np.amin([agent.sz for agent in self.agents])
        self.map_pts = env.map.pts
        self.min_dist_ij = np.ones(len(self.agents))
        pass

    def update(self):
        self.agents_loc = []
        for a in self.env.agents:
            x, y, a = a.state
            self.agents_loc.append([int(x//self.cell_sz), int(y//self.cell_sz)])
        self.agents_loc = np.array(self.agents_loc)

    def detect(self):
        self.update()
        m_pts = self.map_pts# - np.array([env.metadata['world_width']//2, env.metadata['world_height']//2])
        agent_collision_vecs = np.zeros([len(self.agents), len(self.agents), 2])
        wall_collision_vecs = np.zeros([len(self.agents), 2])
        for i, a_loc in enumerate(self.agents_loc):
            a_loc = self.agents_loc[i]
            i_pts = self.agents[i].pts + self.agents[i].state[:2] + self.agents[i].v * self.env.dt * self.env.scale
            for j, b_loc in enumerate(self.agents_loc[:i]):
                if i == j:
                    continue
                # if two agents are in the neighboring cells
                if np.amax(np.abs(a_loc - b_loc)) <= 1:
                    j_pts = self.agents[j].pts + self.agents[j].state[:2] + self.agents[j].v * self.env.dt * self.env.scale
                    dist_ij = cdist(i_pts, j_pts, 'euclidean')
                    if np.amin(dist_ij) < self.eps:
                        if collision_info:
                            print('Collission detected between agent {} and {}'.format(i, j))
                        loc_i = np.array(self.agents[i].state[:2])
                        loc_j = np.array(self.agents[j].state[:2])
                        vec= np.array((loc_j - loc_i) / norm(loc_j - loc_i))
                        agent_collision_vecs[i, j] = vec / norm(vec)
                        self.agents[i].v = (self.agents[i].v - vec * norm(self.agents[i].v))*rebounce
                        self.agents[j].v = (self.agents[j].v + vec * norm(self.agents[j].v))*rebounce
                    # collision between agents and map

            dist_im = cdist(i_pts, m_pts)
            if np.amin(dist_im) < 2.0:
                if collision_info:
                    print('Collision detected btween agent {} and the map'.format(i))
                x, y = np.where(dist_im == np.amin(dist_im))
                pi = np.array(self.agents[i].state[:2])
                pm = self.map_pts[y[0]]
                vec = np.array((pm - pi) / norm(pm - pi))
                self.agents[i].v = (- vec * norm(self.agents[i].v))*rebounce
                wall_collision_vecs[i] = vec
        return agent_collision_vecs, wall_collision_vecs
    #
    # def correct(self):
    #     for i, agent in enumerate(self.env.agents):
    #         v = agent.v
    #         collision_vecs = self.detect(i, v)
    #         if len(collision_vecs):
    #             for vec in collision_vecs:
    #                 if np.inner(v, vec) >= 0:
    #                     agent.v = np.array([0, 0])
    #                 #v -= vec / norm(vec)**2 * np.inner(v, vec) * rebounce

def rot_mat(vec, a):
    return np.array([np.cos(a) * vec[0] - np.sin(a) * vec[1], np.sin(a) * vec[0] + np.cos(a) * vec[1]])

def clip(vec, max_norm=1.5):
    if norm(vec) > max_norm:
        return vec / norm(vec) * max_norm
    return vec

class Map(Geom2d):
    def __init__(self):
        self.pts = []

    def get_map_from_bitmap(self):
        return

    def get_map_from_geom2d(self, env, radius=100, geom_type='circle', color=(0.0, 0.0, 0.0, 1), parent=None, n_pts=100):
        kp = np.array([[-radius, 0], [radius, 0]])
        super().__init__(env=env, kp=kp, geom_type=geom_type, filled=False, color=color, parent=None, n_pts=n_pts)
        self.geom = super()._render()
        env.map = self

    def _render(self):
        return self.geom

    def get_map_from_kps(self):
        pass
# class Target(Agent):
#     metadata = {
#         'vel_max': np.random.rand() * 1.5
#     }
#     def __init__(self):
#         super().__init__()
#         self.vel_max = self.metadata['vel_max']
#         self.vel = clip(np.random.rand(2), max_norm=self.vel_max)
#     def update(self):
#         if np.random.rand() < 0.05:
#             self.vel
#         return

if __name__ == '__main__':
    env = Simulator()
    map = Map()
    map.get_map_from_geom2d(env, radius=100, n_pts=180)


    robot = Agent(env, radius=3, color=(1,0,0,0.5), v_max=1.5)
    blob_finder =Blob_finder(env, range=30, color=[0, 1, 0, 0.1])
    # robot.add_device(blob_finder)
    while True:
        env._render()
        robot.update(v=np.array([1.0, 0]))

    # # ---------------------------
    # #    6 robots and 6 targets
    # # ---------------------------
    # #
    # n_targets = 5
    # n_robots = 5
    #
    # targets = [Agent(env, radius=1, color=(1,0,0,0.5), v_max=1.5) for i in range(n_targets)]
    # robots = [Agent(env, radius=2, color=(0,0,1,0.5), v_max=2) for i in range(n_targets, n_robots+n_targets)]
    # tic = time()
    # vs = (np.random.rand(n_targets+n_robots,2) - 0.5) * 4
    #
    #
    # while True:
    #     env._render()
    #     for i in range(n_targets + n_robots):
    #         if np.random.rand() < 0.02:
    #             vs[i] = (np.random.rand(2) - 0.5) * 4
    #         else:
    #             vs[i] = vs[i]
    #     for i, t in enumerate(targets+robots):
    #         t.update(vs[i])
    #     # print('update takes {} sec'.format(time() - tic))
    #     # print()
    #     tic = time()
