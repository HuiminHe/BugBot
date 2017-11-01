import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
import random
from rendering import Geom2d, Viewer, Transform
import simulator_config
from datetime import datetime

simulator_config.collision_info = False


class Simulator(object):

    def __init__(self, config):
        self.config = config
        self.viewer = None
        self.state = None
        self.map = None
        self.agents = []
        self.dt = self.config.metadata['dt']
        self.effects = []
        self.markers = []
        self.scale = np.ceil(self.config.metadata['screen_width'] / self.config.metadata['world_width'])
        self.config.metadata['screen_width'] = int(self.config.metadata['world_width'] * self.scale)
        self.move_to_center = Transform(translation=(self.config.metadata['screen_width'] // 2, self.config.metadata['screen_height'] // 2))
        self.counter = 0

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = Viewer(self.config.metadata['screen_width'], self.config.metadata['screen_height'])
            if self.map:
                geom = self.map._render()
                geom.add_attr(self.move_to_center)
                self.viewer.add_geom(geom)

            if self.markers:
                for m in self.markers:
                    m.add_attr(self.move_to_center)
                    self.viewer.add_geom(m)
                    
            if len(self.agents):
                for i, agent in enumerate(self.agents):
                    # agent.reset(init_state=np.array([x[i],y[i], 0.0]))
                    geom = agent._render()
                    self.viewer.add_geom(geom)

                self.collision = CollisionDetector(self)

        if len(self.agents):
            # update agents motion
            self.collision.detect()
            for agent in self.agents:
                agent._update_render()
                for d in agent.devices:
                    geom = d._render()
                    self.viewer.add_onetime(geom)

        for e in self.effects:
            self.viewer.add_onetime(e)

        # update collision detector
        self.counter += 1
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def add_effect(self, effect):
        '''
        add a one-time geometry in the simulation not detected by collision detector
        '''
        self.effects.append(effect)

    def add_marker(self, marker):
        '''
        add a geometry that is not detected by collision detector
        '''
        self.markers.append(marker)
        
class Agent(Geom2d):
    counter = 0

    def __init__(self, env, kp, device=None, color=(1,0,0,0.5), v_max=2.0, a_max=99):
        super().__init__(env, kp=kp, color=color, parent=None)
        env.agents.append(self)
        self.indx = type(self).counter
        type(self).counter += 1
        self.geom = super()._render()
        self.devices = []
        self.color = color
        self.rot = Transform()
        self.mov = Transform()
        self.trans = [self.rot, self.mov]
        self.v_max = v_max
        self.a_max = a_max

        self.geom.add_attr(self.rot)
        self.geom.add_attr(self.mov)
        self.geom.add_attr(self.env.move_to_center)
        self.reset()

    def _render(self) :
            return self.geom

    def _update_render(self):
        if not equal(self.ac, self.ac_last) and not equal(self.v, self.v_last + self.ac_last * self.env.dt):
            if self.inCollision or self.initializing:
                pass
            else:
                raise ValueError('User can only set velocity or acceleration')

        if self.initializing:
            self.initializing = False


        # velocity control
        if not equal(self.v, self.v_last + self.ac_last * self.env.dt):
            self.ac = clip(self.v - self.v_last, self.a_max)
            self.v = self.v_last + self.ac * self.env.dt
        else:
            self.ac = clip(self.ac, self.a_max)
        self.v = self.v_last + self.ac * self.env.dt

        # clip vel
        if norm(self.v) > self.v_max:
            self.v = clip(self.v, self.v_max)

        # update state
        x, y, a = self.state
        x += self.v[0] * self.env.dt
        y += self.v[1] * self.env.dt
        a += self.va * self.env.dt
        self.state = np.array([x, y, a])
        self.rot.set_rotation(a * self.env.dt)
        self.mov.set_translation(x * self.env.scale, y * self.env.scale)
        self.v_last = np.array(self.v)
        self.ac_last = np.array(self.ac)
        
    def reset(self, init_state=()):
        self.init_state = init_state
        if len(init_state):
            self.state = init_state
        else:
            self.state =  (np.random.rand(3) - 0.5) * self.env.map.sz * 0.7 / 2
        self.v = rot_mat(np.array([self.v_max, 0.0]), np.random.rand()*np.pi*2)
        self.v_last = np.array([0, 0])
        self.va = 0 # angular velocity
        self.ac = np.array([0 ,0]) # acceleration
        self.ac_last = np.array([0, 0])
        self.aa = 0.0 #angular acceleration
        self.inCollision = False
        self.initializing = True

    def update(self, v=np.array([0.0, 0.0]), va=0):
        v = clip(v, max_norm=self.v_max)
        self.v = v
        self.va = va

    def loc(self):
        return self.state[:2]

class CollisionDetector(object):
    def __init__(self, env):
        self.env = env
        self.eps = env.config.metadata['eps']
        self.agents = env.agents
        self.cell_sz = np.amin([agent.sz for agent in self.agents])
        self.map_pts = env.map.pts
        self.min_dist_ij = np.ones(len(self.agents))
        pass

    def update(self):
        self.agents_loc = []
        for a in self.env.agents:
            x, y, a = a.state
            self.agents_loc.append([int(x/self.cell_sz), int(y/self.cell_sz)])
        self.agents_loc = np.array(self.agents_loc)

    def detect(self):
        self.update()
        m_pts = self.map_pts# - np.array([env.config.metadata['world_width']//2, env.config.metadata['world_height']//2])
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
                        if simulator_config.collision_info:
                            print('Collission detected between agent {} and {}'.format(i, j))
                        loc_i = np.array(self.agents[i].state[:2])
                        loc_j = np.array(self.agents[j].state[:2])
                        vec= np.array((loc_j - loc_i) / norm(loc_j - loc_i))
                        agent_collision_vecs[i, j] = vec / norm(vec)
                        self.agents[i].v -= (vec * norm(self.agents[i].v))*(1.1 + self.env.config.rebounce)
                        self.agents[j].v += (vec * norm(self.agents[j].v))*(1.1 + self.env.config.rebounce)
                        self.agents[i].inCollision = True
                        self.agents[j].inCollision = True

            # collision between agents and map
            dist_im = cdist(i_pts, m_pts)
            if np.amin(dist_im) < simulator_config.metadata['eps']:
                if simulator_config.collision_info:
                    print('Collision detected btween agent {} and the map'.format(i))
                x, y = np.where(dist_im == np.amin(dist_im))
                pi = np.array(self.agents[i].state[:2])
                pm = self.map_pts[y[0]]
                vec = np.array((pm - pi) / norm(pm - pi))
                self.agents[i].v = (- vec * norm(self.agents[i].v))*(1.1 + self.env.config.rebounce)
                wall_collision_vecs[i] = vec
                self.agents[i].inCollision = True
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
    #                 #v -= vec / norm(vec)**2 * np.inner(v, vec) * (1.1 + self.env.config.rebounce)

def rot_mat(vec, a):
    return np.array([np.cos(a) * vec[0] - np.sin(a) * vec[1], np.sin(a) * vec[0] + np.cos(a) * vec[1]])

def dire(vec):
    return vec / norm(vec)

def clip(vec, max_norm=1.5):
    if norm(vec) > max_norm:
        return dire(vec) * max_norm
    return vec

def equal(vec1, vec2):
    if len(vec1) != len(vec2):
        return False
    elif norm(vec1 - vec2) == 0:
        return True
    else:
        return False

class Map(Geom2d):
    def __init__(self):
        pass

    def get_map_from_bitmap(self):
        return

    def get_map_from_geom2d(self, env, kp, color=(0.0, 0.0, 0.0, 1), parent=None):
        super().__init__(env=env, kp=kp, filled=False, color=color, parent=None)
        self.geom = super()._render()
        env.map = self

    def _render(self):
        return self.geom

    def get_map_from_kps(self):
        pass


if __name__ == '__main__':
    pass
