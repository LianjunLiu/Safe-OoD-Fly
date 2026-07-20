import re
import json
import rowan
import numpy as np
from collections import namedtuple


class DualOutput:
    """Output simultaneously to the file and the terminal"""
    def __init__(self, file_handle, terminal):
        self.file = file_handle
        self.terminal = terminal
    
    def write(self, message):
        self.file.write(message)
        self.terminal.write(message)
    
    def flush(self):
        self.file.flush()
        self.terminal.flush()


# MATLAB struct field names cannot contain hyphens or begin with digits.
def matlab_safe_name(name: str) -> str:
    safe = re.sub(r'\W', '_', name)
    if safe and safe[0].isdigit():
        safe = f'_{safe}'
    return safe


def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params


def format_plot(ax):
    ax.margins(x=0)


def get_subclass_list(cls):
    names = []
    for c in cls.__subclasses__():
        names.append(c._name)
    return names


def get_subclass(cls, name, **kwargs):
    names = get_subclass_list(cls)
    for c in cls.__subclasses__():
        if name == c._name or name in c._names:
            return c(**kwargs)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getattr__(self, key):
        return self[key]


_nan_array = np.empty((3,3))
_nan_array[:] = np.nan
_nan_array.flags.writeable = False
State = namedtuple('State', 'p q v w R', defaults=(_nan_array))

def get_state_components(X):
    return State(p=X[0:3], q=X[3:7], v=X[7:10], w=X[10:])

def get_state_components_with_R(X):
    return State(p=X[0:3], q=X[3:7], v=X[7:10], w=X[10:], R=rowan.to_matrix(X[3:7]))


Statistics = namedtuple('Statistics', 'count mean rmse max std')

class StatisticsTracker():
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.err_mean = None
        self.err_squ_cum = None
        self.err_max = None
        self.err_var = None
    
    def update(self, err):
        self.count += 1
        if self.err_mean is None:
            self.err_mean = np.zeros_like(err)
            self.err_max = np.zeros_like(err)
            self.err_squ_cum = np.zeros_like(err)
            self.err_var = np.zeros_like(err)
        self.err_max = np.maximum(err, self.err_max)
        self.err_squ_cum = self.err_squ_cum + err ** 2
        err_mean_old = self.err_mean
        self.err_mean = self.err_mean + (err - self.err_mean) / self.count
        self.err_var = self.err_var + err_mean_old ** 2 - self.err_mean ** 2 + (err ** 2 - self.err_var - err_mean_old ** 2) / self.count

    def get_statistics(self):
        if self.count > 0:
            return Statistics(self.count, self.err_mean, np.sqrt(self.err_squ_cum / self.count), self.err_max, np.sqrt(self.err_var))
        else:
            return Statistics(0.,0.,0.,0.,0.)
