import os
import numpy as np
import matplotlib.pyplot as plt


def get_position_error(X, pd, istart=0, iend=-1, lower_percentile=25, upper_percentile=75):
    istart = int(istart)
    iend = int(iend)

    squ_error = np.sum((X[istart:, 0:3] - pd[istart:])**2,1)
    rmse = np.sqrt(np.mean(squ_error))

    meanerr = np.mean(np.sqrt(squ_error))
    maxerr = np.max(np.sqrt(squ_error))

    fifth = np.sqrt(np.percentile(squ_error, lower_percentile))
    ninetyfifth = np.sqrt(np.percentile(squ_error, upper_percentile))
    
    return dict(rmse=rmse, fifth=fifth, ninetyfifth=ninetyfifth, meanerr=meanerr, maxerr=maxerr)

def get_average_control_error(X, pd, vd, istart=0, iend=-1, lower_percentile=25, upper_percentile=75):
    istart = int(istart)
    iend = int(iend)

    squ_error = np.sum((X[istart:, [0, 1, 2, 7, 8, 9]] - np.concatenate((pd[istart:], vd[istart:]),axis=1))**2,1)
    rmse = np.sqrt(np.mean(squ_error))

    meanerr = np.mean(np.sqrt(squ_error))
    maxerr = np.max(np.sqrt(squ_error))

    fifth = np.sqrt(np.percentile(squ_error, lower_percentile))
    ninetyfifth = np.sqrt(np.percentile(squ_error, upper_percentile))
    
    return dict(rmse=rmse, fifth=fifth, ninetyfifth=ninetyfifth, meanerr=meanerr, maxerr=maxerr)

def plot_3d(log, bound=2.5, savename=None, nametag='Quadrotor simulation'):
    fig = plt.figure(figsize=(6,3))

    fig.add_subplot(121, projection='3d')
    plt.plot(log['X'][:,0],log['X'][:,1], log['X'][:,2])
    plt.plot(log['pd'][:,0], log['pd'][:,1], log['pd'][:,2])
    ax = plt.gca()
    ax.set_xlim(-bound,bound)
    ax.set_ylim(-bound,bound)
    ax.set_zlim(-bound,bound)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    fig.add_subplot(122)
    plt.plot(log['X'][:,0],log['X'][:,2])
    plt.plot(log['pd'][:,0], log['pd'][:,2])
    plt.axis((-bound,bound,-bound,bound))
    plt.xlabel('x')
    plt.ylabel('z')

    rmse = np.sqrt(np.mean(np.sum((log['X'][:, 0:3] - log['pd'][:])**2,1)))
    plt.title(nametag + '\nrmse = ' + '%.3fm' % rmse)
    
    if savename is not None:
        plt.savefig(savename)

def plot_xyz(log, savename=None):
    plt.figure(figsize=(9,3))
    plt.title('Actual vs. desired position')

    plt.subplot(3,1,1)
    plt.plot(log['t'], log['X'][:,0])
    plt.plot(log['t'], log['pd'][:,0])
    plt.legend(('x act', 'x des',))
    plt.xlabel('t')
    plt.ylabel('x')

    plt.subplot(3,1,2)
    plt.plot(log['t'], log['X'][:,1])
    plt.plot(log['t'], log['pd'][:,1])
    plt.legend(( 'y act', 'y des'))
    plt.xlabel('t')
    plt.ylabel('y')

    plt.subplot(3,1,3)
    plt.plot(log['t'], log['X'][:,2])
    plt.plot(log['t'], log['pd'][:,2])
    plt.legend(( 'z act', 'z des'))
    plt.xlabel('t')
    plt.ylabel('z')

    if savename is not None:
        plt.savefig(savename)

def plot_error(log, istart=0):
    plt.figure()
    plt.plot(log['t'][istart:], np.sum((log['X'][istart:, 0:3] - log['pd'][istart:,:])**2,1))

def plot_savefig(plottag):
    from pathlib import Path
    savepath = str(Path(plottag + '.pdf').resolve())
    if os.name == 'nt' and len(savepath) > 200:
        savepath = '\\\\?\\' + savepath
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight')
