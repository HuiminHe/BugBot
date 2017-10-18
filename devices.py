from rendering import Geom2d, Transform
import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import norm
class Device(Geom2d):
    def __init__(self, env, parent, kp=np.array([[-1, 0], [1, 0]]), color=(1,0,0,0.1)):
        self.env = env
        self.bd = kp
        self.geom = None
        self.color = color
        self.parent = parent
        super().__init__(env=self.env, kp=self.bd, geom_type='polygon', color=self.color, parent=parent)


    def _render(self):
        self.geom = super()._render()
        self.attach = Transform(translation=(self.parent.sz * self.env.scale // 5, self.parent.sz * self.env.scale //3))
        self.geom.add_attr(self.attach)
        self.geom.add_attr(self.parent.mov)
        self.geom.add_attr(self.parent.rot)
        return self.geom

def get_theta(c, pt):
    theta = np.arctan2(pt[1] - c[1], pt[0] - c[0])
    return np.rad2deg(theta if theta > 0 else theta + np.pi).astype(np.int16)

def get_thetas(c, pts):
    theta = []
    for pt in pts:
        theta.append(get_theta(c, pt))
    return np.array(theta)

def dist(p1, p2):
    return norm(p1 - p2)

class Blob_finder(Device):
    def __init__(self, env, parent, range=10, res=30, color=(1, 0, 0, 0.1)):
        self.env = env
        self.geom_type = 'polygon'
        self.filled = True
        self.color = color
        self.trans = []
        self.range = range
        self.parent = parent
        self.sz = self.range * 2
        self.res = res # resolution
        self.kp = [np.array([np.cos(th), np.sin(th)]) * self.range for th in np.arange(self.res) * (360//self.res)]


    def _read(self):
        cell2check = np.ceil(self.sz / self.env.collision.cell_sz)
        a_loc = self.env.collision.agents_loc[self.parent.indx]
        i = self.parent.indx
        c = self.parent.state[:2]
        kp = [np.array([np.cos(th), np.sin(th)]) * self.range + c for th in np.arange(self.res) * (360//self.res)]

        pts = self.env.map.pts
        dis = cdist([c], pts)
        print(np.amin(dis))
        if np.amin(dis) < self.range:
            index = np.where( dis < self.range)[1].astype(np.int16)
            theta_indx = (get_thetas(c, self.env.map.pts[index]) / (360/self.res)).astype(np.int16)
            for i in range(len(index)):
                kp[theta_indx[i]] = pts[index[i]]

        super().__init__(env=self.env, parent=self.parent, kp=kp, color=self.color)
        # for j, b_loc in enumerate(self.env.collision.agents_loc):
        #     if i == j:
        #         continue
        #     if np.amax(np.abs(a_loc - b_loc)) <= cell2check:
        #         pts = self.env.agents[j].pts
        #         indx = np.where(cdist(c, pts))
        #         theta_indx = get_thetas(c, env.agents[j].pts[indx])
        #         kp = [pts[ind] if dist(c, pts[ind]) < dist(c, kp[th_ind]) else kp[theta_in] for ind, th_ind in zip(indx, theta_indx)]
        return np.array(kp)
