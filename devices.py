from rendering import Geom2d, Transform
import numpy as np


class Device(Geom2d):
    def __init__(self, env, parent, kp=np.array([[-1, 0], [1, 0]]), color=(1,0,0,0.5)):
        self.env = env
        self.bd = kp
        self.geom = None
        self.color = color
        self.parent = parent
        self.parent.devices.append(self)

        if len(kp) > 2:
            super().__init__(env=self.env, kp=self.bd, geom_type='polygon', color=self.color, parent=parent)
        elif len(kp) == 2:
            super().__init__(env=self.env, kp=self.bd, geom_type='circle', color=self.color, parent=parent)

        self.geom = super(Device, self)._render()
        self.geom.add_attr(self.parent.rot)
        self.geom.add_attr(self.parent.mov)

    def _render(self):
        return self.geom
