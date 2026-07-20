import random
import numpy as np
from matplotlib import pyplot as plt

def setpoint(pd):
    vd = ad = jd = sd = np.zeros(pd.shape)
    return pd, vd, ad, jd, sd


class Trajectory():
    _name = None
    _names = []

    def __init__(self):
        raise NotImplementedError

    def __call__(self, t):
        raise NotImplementedError


class Hover(Trajectory):
    _name = 'Hover'
    _names = []

    def __init__(self, pd=np.zeros(3)):
        self.pd = pd

    def __call__(self, t):
        return setpoint(self.pd)


class RandomPoint(Trajectory):
    _name = 'RandomPoint'
    _names = []

    def __init__(self, t0=0., T=2.0, bounds=None, seed=None):
        if bounds is None:
            self.xmin = -2.5
            self.xmax = 2.5
            self.ymin = -1.5
            self.ymax = 1.5
            self.zmin = -1.5
            self.zmax = 1.5
        else:
            self.xmin = bounds['xmin']
            self.xmax = bounds['xmax']
            self.ymin = bounds['ymin']
            self.ymax = bounds['ymax']
            self.zmin = bounds['zmin']
            self.zmax = bounds['zmax']

        self.t_lastupdate = t0
        self.T = T
        self.pd = np.zeros(3)

        self.metadata = {}
        self.metadata['trajectory'] = 'random setpoint'
        
        if seed is not None:
            random.seed(seed)

    def __call__(self, t):
        if t > self.t_lastupdate + self.T:
            self.pd = np.array([random.uniform(self.xmin, self.xmax), random.uniform(self.ymin, self.ymax), random.uniform(self.zmin, self.zmax)])
            self.t_lastupdate = t
        return setpoint(self.pd)


class Fig8(Trajectory):
    _name = 'Fig8'
    _names = []

    def __init__(self, T=6.0, dir1=(2., 2., 0.), dir2=(0., 0., 1.)):
        self.w = 2 * np.pi / T
        self.dir1 = np.array(dir1)
        self.dir2 = np.array(dir2)

    def __call__(self, t):
        pd =                np.sin(self.w*t) * self.dir1 +                 np.sin(2*self.w*t) * self.dir2
        vd =   self.w     * np.cos(self.w*t) * self.dir1 +  2*self.w     * np.cos(2*self.w*t) * self.dir2
        ad = -(self.w)**2 * np.sin(self.w*t) * self.dir1 - (2*self.w)**2 * np.sin(2*self.w*t) * self.dir2
        jd = -(self.w)**3 * np.cos(self.w*t) * self.dir1 - (2*self.w)**3 * np.cos(2*self.w*t) * self.dir2
        sd =  (self.w)**4 * np.sin(self.w*t) * self.dir1 + (2*self.w)**4 * np.sin(2*self.w*t) * self.dir2
        return pd, vd, ad, jd, sd


class SinForward():
    _name = 'SinForward'
    _names = []

    def __init__(self, T=6., A=2, Vy=0.2, Vz=0.5):
        self.w = np.pi * 2/T
        self.A = A
        self.Vy = Vy
        self.Vz = Vz
    
    def __call__(self,t):
        pd = np.array((self.A*np.sin(self.w*t), self.Vy*t, self.Vz*t))
        vd = np.array((self.w * self.A * np.cos(self.w*t), self.Vy, self.Vz))
        ad = np.array((-self.w**2 * self.A * np.sin(self.w*t), 0, 0))
        jd = np.array((-self.w**3 * self.A * np.cos(self.w*t), 0, 0))
        sd = np.array((self.w**4 * self.A * np.sin(self.w*t), 0, 0))
        
        return pd, vd, ad, jd, sd


class SpiralUp():
    _name = 'SpiralUp'
    _names = []

    def __init__(self, T=5., R=2.0, Vz=0.4, Vr=0.2):
        self.w = np.pi*2/T
        self.R = R
        self.Vr = Vr
        self.Vz = Vz
    
    def __call__(self, t):
        pd = self.R*np.array((np.sin(self.w*t), np.cos(self.w*t)-1, self.Vz*t/self.R))
        vd = self.R*np.array((self.w * np.cos(self.w*t), -self.w * np.sin(self.w*t), self.Vz/self.R))
        ad = self.R*np.array((-self.w**2 * np.sin(self.w*t), -self.w**2 * np.cos(self.w*t), 0))
        jd = self.R*np.array((-self.w**3 * np.cos(self.w*t), self.w**3 * np.sin(self.w*t), 0))
        sd = self.R*np.array((self.w**4 * np.sin(self.w*t), self.w**4 * np.cos(self.w*t), 0))
        
        return pd, vd, ad, jd, sd


class HourGlass(Trajectory):
    _name = 'HourGlass'
    _names = []

    def __init__(self, t0=0., time_scale=1., vertices=((0., 0., 0.), (1., 0., 1.), (1., 0., 1.), (-1., 0., 1.), (-1., 0., 1.), (1., 0., -1.), (1., 0., -1.), (-1., 0., -1.), (-1., 0., -1.)), loop_start_idx=1):
        
        self.time_scale = time_scale
        self.vertices = np.array(vertices)

        dist_next = np.linalg.norm(self.vertices[1] - self.vertices[0])
        self.T_to_next = self.time_scale * dist_next

        self.next_vertex = 0
        self.loop_start_idx = loop_start_idx
        self.t_last_vertex = t0 - self.T_to_next

        self.p0 = self.vertices[0]
        self.p1 = self.vertices[0]

        self._e = 35
        self._f = -84
        self._g = 70
        self._h = -20
        
        self.metadata = {}
        self.metadata['trajectory'] = 'random setpoint'

    def _T(self, t, tf):
        tt = t/tf
        return self._e * tt**4 + self._f * tt**5 + self._g * tt**6 + self._h * tt**7

    def _Tdot(self, t, tf):
        tt = t/tf
        return (4 / tf) * self._e * tt**3 + (5 / tf) * self._f * tt**4 + (6 / tf) * self._g * tt**5 + (7 / tf) * self._h * tt**6
            
    def _Tddot(self, t, tf):
        tt = t/tf
        return (3 / tf) * (4 / tf) * self._e * tt**2 + (4 / tf) * (5 / tf) * self._f * tt**3 + (5 / tf) * (6 / tf) * self._g * tt**4 + (6 / tf) * (7 / tf) * self._h * tt**5
        
    def __call__(self, t):
        if t > self.t_last_vertex + self.T_to_next:
            last_vertex = self.vertices[self.next_vertex]
            self.next_vertex += 1
            if self.next_vertex >= self.vertices.shape[0]:
                self.next_vertex = self.loop_start_idx
            self.t_last_vertex = self.t_last_vertex + self.T_to_next
            dist_next = np.linalg.norm(last_vertex - self.vertices[self.next_vertex])
            if dist_next < 0.1:
                dist_next = 1.
            self.T_to_next = self.time_scale * dist_next
            self.p0 = self.p1
            self.p1 = self.vertices[self.next_vertex]
        p = self.p0 + (self.p1 - self.p0) * self._T(t - self.t_last_vertex, self.T_to_next)
        v = (self.p1 - self.p0) * self._Tdot(t - self.t_last_vertex, self.T_to_next)
        a = (self.p1 - self.p0) * self._Tddot(t - self.t_last_vertex, self.T_to_next)
        j = np.inf * np.ones(3)
        s = np.inf * np.ones(3)
        return p, v, a, j, s


def get_trajectory(name, **kwargs):
    for t in Trajectory.__subclasses__():
        if name == t._name or name in t._names:
            return t(**kwargs)


# trajectory_names = []
# for t in Trajectory.__subclasses__():
#     trajectory_names.append(t._name)
# print("Registered trajectories: ", trajectory_names)


# Example usage
if __name__ == "__main__":
    traj = Fig8(T=10.0)
    times = np.linspace(0, 20, 500)
    positions = []
    for t in times:
        pd, vd, ad, jd, sd = traj(t)
        positions.append(pd)
    positions = np.array(positions)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_title('Fig8 Trajectory')
    plt.show()
