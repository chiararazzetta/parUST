# %%
import ctypes
from numpy.ctypeslib import load_library
from numpy.ctypeslib import ndpointer
import math
import numpy as np
import scipy
from multiprocessing import cpu_count
from joblib import Parallel, delayed

from f_initialization import rit_g, gridcreate, element_discr
# %%
risplib = load_library('matrix.so', '.')

H = risplib.hmat

H.restype = ndpointer(ctypes.c_float)
H.argtypes = [ndpointer(ctypes.c_float),
              ctypes.c_int,
              ndpointer(ctypes.c_float),
              ctypes.c_int,
              ctypes.c_float,
              ctypes.c_float,
              ndpointer(ctypes.c_float),
              ctypes.c_int,
              ndpointer(ctypes.c_float),
              ndpointer(ctypes.c_float)
              ]


def mrisptopy(cen, p, v, dt, g):
    """ Function to link python objects to c function

    Args:
        cen (numpy.float): array containing the coordinates of element discretization points
        p (numpy.float): array containing the coordinates of the points of the field
        v (float):medium speed
        dt (float): time step
        g (numpy.float): array containing the geometric delays to simulate the probe lens

    Returns:
        list : two numpy arrays, the array of impulse response values and the corresponding time
    """
    row = cen.shape[0]
    npunti = p.shape[0]
    h = np.zeros((npunti, 300), dtype=np.float32)
    t = np.zeros(npunti, dtype=np.float32)
    
    H(cen, row, p, npunti, v, dt, g, 1, h, t)
    
    return h/row, t

# %%

def narrowMap(pitch, c, dt, geom, grid, Nx, Nz, idx, cen, f0):
    """ Function to generate a impulse response map for a grid of points and a fixed probe element
        in Narrow Band pulse transmission setting, with mono frequency approximation 

    Args:
        pitch (float): element length for translations on the probe
        c (float): (float):medium speed
        dt (float): time step
        geom (numpy.float): array containing the geometric delays to simulate the probe lens
        grid (numpy.float): array containing the coordinates of the points of the field
        Nx (int): grid size along xaxis
        Nz (int): grid size along zaxis
        idx (int): index of the currently processed element
        cen (numpy.float): array containing the coordinates of element discretization points
        f0 (float): central frequency of the narrow pulse

    Returns:
        list: a numpy array containing the collection of the impulse responses, a tuple containing the grid dimensions
    """
    T = np.arange(0, 300 * dt, dt)
    phase = np.exp(-2 * math.pi * 1j * T * f0)

    grid[:, 0] -= (idx) * pitch
    h, t = mrisptopy(cen, grid, c, dt, geom)

    trif = t - grid[:, 2] / c
    h = np.sum(h * phase, 1)
    H = h * np.exp(-2 * math.pi * 1j * f0 * trif)

    return np.reshape(H, (Nx, Nz))

# %%


def wideMap(c, dt, geom, grid, cen, nt, pad, A, hprobe = None):
    """ Function to generate a impulse response map for a grid of points and a fixed probe element
        in Wide Band pulse transmission setting
    
    Args:
        c (float): medium speed
        dt (float): time step
        geom (numpy.float): array containing the geometric delays to simulate the probe lens
        grid (numpy.float): array containing the coordinates of the points of the field
        cen (numpy.float): array containing the coordinates of element discretization points
        nt (int): maximum number of temporal instant
        pad (int): number of zero for signals padding
        A (numpy.float): array containing the attenuation coefficients
    
    Optional Args:
        hprobe (list): if it is possible to have a measurements of the probe impulse response, we cut the significant frequencies
                       and take in account its shape

    Returns:
        numpy.complex: array containing the collection of the impulse responses
    """       

    h, t = mrisptopy(cen, grid, c, dt, geom, nt)
    h = np.pad(h, ((0, 0), (pad, 0)))

    h = scipy.fft.fft(h, axis=1)
    h = h[:, :(nt + pad) // 2]

    trif = t - grid[:, 2] / c

    freq = scipy.fft.fftfreq(nt + pad, dt)[:(nt + pad) // 2]
    freq = np.repeat(freq.reshape([1, (nt + pad) // 2]), trif.shape, axis=0)

    if hprobe is None:
        phase = np.exp(-2 * math.pi * 1j *freq * trif[:, np.newaxis])
        h = h * phase  * A
    else: 
        dim = hprobe[0]
        hp = hprobe[1]
        phase = np.exp(-2 * math.pi * 1j *freq[:, dim[0]:dim[1]] * trif[:, np.newaxis])
        h = h[:, dim[0]:dim[1]] * phase * hp[np.newaxis, :] * A[:, dim[0]:dim[1]]

    return h

# %%
def Narrow_att_map(coordz, f0, factor, N):
    """ Function for computing the attenuation map in single frequency Narrow Band case

    Args:
        coordz (numpy.float): array containing all the possible depths in the field
        f0 (float): central frequency
        factor (float): factor for attenuation rule (0.5 for narrow transmission 0.3 for wide transmission)
        N (int): number of point along the xaxis

    Returns:
        numpy.complex: array containing the attenuation map with grid size
    """
    
    attmap = 10 ** ((-factor/20) * (f0 * coordz)/(10 ** 4))

    return np.tile(attmap, (N, 1))

#%% 

def Wide_att_map(coordz, Nz, Nx, factor, nt, pad, dt):
    """ Function for computing the attenuation map in Wide Band case

    Args:
        coordz (numpy.float): array containing all the possible depths in the field
        Nz (int): number of point along the zaxis
        Nx (int): number of point along the xaxis
        factor (float): factor for attenuation rule
        nt (int): maximum number of temporal instant
        pad (int): number of zerod for signals padding
        dt (float): time step

    Returns:
        numpy.complex: array containing the attenuation map with grid size
    """    
    w = np.linspace(0, 1/dt, pad + nt + 1)
    wreal = np.abs(w[0 : (nt + pad) // 2])

    attmap = np.empty([Nz, (nt + pad) // 2])

    for i in range(Nz):
        attmap[i, :] = 10 ** (-factor * coordz[i] * wreal / (10 ** 5))

    return np.tile(attmap, (Nx,1))

# %%

def parallelMapcompute(pitch, c, dt, geom, grid, nel, step, Nx, Nz, cen, f0, min_d, max_d, i):
    """
        Auxiliar function for parallelizing maps computing
    """
    grid, indexes, Nx = gridcreate(pitch, nel, step, Nz, min_d, max_d)
    H1 = narrowMap(pitch, c, dt, geom, grid, Nx, Nz, i + 0.5, cen, f0)
    grid, indexes, Nx = gridcreate(pitch, nel, step, Nz, min_d, max_d)
    H2 = narrowMap(pitch, c, dt, geom, grid, Nx, Nz, -i-1+0.5, cen, f0)
    return H1 + H2

def NarrowMaps(pitch, cen, f_g, nel, c, dt, step, Nz, min_d, max_d, factor, f0):
    """Function for the parallel computing of Narrow maps for multiple element
    
    Args:
        pitch (float): element length for translations on the probe
        cen (numpy.float): array containing the coordinates of element discretization points
        f_g (float): geometrical focus of the probe
        nel (int): number of elements of the probe to be considered
        c (float): medium speed
        dt (float): time step
        step (float): fractional part of the probe element for spacing the x coordinates
        Nz (int): number of points along depth
        min_d (float): minimum depth
        max_d (float): maximum depth
        factor (float): attenuation factor
        f0 (float): transmission frequency

    Returns:
        numpy.complex: the maps array
        numpy.complex: the attenuation map
        numpy.grid: the grid coordinates
        list: the grid dimesions
    """    
    geom = rit_g(f_g, cen[:,1], c)

    grid, indexes, Nx = gridcreate(pitch, nel, step, Nz, min_d, max_d)

    A = Narrow_att_map(grid[:Nz,2], f0, factor, Nx)

    H = Parallel(n_jobs = int(cpu_count()-10), backend = "threading")(
                delayed(parallelMapcompute)
                (pitch, c, dt, geom, grid, nel, step, Nx, Nz, cen, f0, min_d, max_d, i) for i in range(nel))
    return np.asarray(H), A, grid, [Nx, Nz]
