from rendering import Geom2d, Transform
import numpy as np
from utils import dist

class Device(Geom2d):
    def __init__(self, env, parent, kp=np.array([[-1, 0], [1, 0]]), color=(1,0,0,0.5), geom_type=None, filled=True):
        self.env = env
        self.bd = kp
        self.geom = None
        self.color = color
        self.parent = parent
        self.parent.devices.append(self)

        if len(kp) > 2 or geom_type=='polygon':
            super().__init__(env=self.env, kp=self.bd, color=self.color, parent=parent, filled=filled)
        elif len(kp) == 2 or geom_type=='circle':
            super().__init__(env=self.env, kp=self.bd, color=self.color, parent=parent, filled=filled)

        self.geom = super()._render()
        self.geom.add_attr(self.env.move_to_center)

    def _render(self):
        return self.geom

class BlobFinder(Device):
    def __init__(self, env, parent, radius, color=(1,0,0,0.5), geom_type=None, filled=True):
        self.radius = radius
        kp=np.array([[-radius, 0], [radius, 0]])
        Device.__init__(self, env, parent, kp=kp, color=color, geom_type=geom_type, filled=filled)

    def read(self):
        blob = []
        for a in self.env.agents:
            if a != self.parent and dist(self.parent, a) < self.radius:
                blob.append({'pos2d':a.loc() - self.parent.loc(), 'color':a.color, 'dist': dist(self.parent, a) - self.parent.sz/2 - a.sz/2})
        return blob
                
