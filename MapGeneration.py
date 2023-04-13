# %%
import ctypes
from numpy.ctypeslib import load_library
from numpy.ctypeslib import ndpointer
import math
import numpy as np
import scipy
import time

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

    print('s')
    tempo = time.time()
    H(cen, row, p, npunti, v, dt, g, 1, h, t)
    print('f')
    print(time.time() - tempo)
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
        list: a numpy array containing the collection of the impulse responses, a tuple containing the grid dimensions
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
        numpy.float: array containing the attenuation map with grid size
    """
    
    attmap = 10 ** ((-factor/20) * (f0 * coordz)/(10 ** 4))

    return attmap.repeat(N, 1)

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
        numpy.float: array containing the attenuation map with grid size
    """    
    w = np.linspace(0, 1/dt, pad + nt + 1)
    wreal = np.abs(w[0 : (nt + pad) // 2])

    attmap = np.empty([Nz, (nt + pad) // 2])

    for i in range(Nz):
        attmap[i, :] = 10 ** (-factor * coordz[i] * wreal / (10 ** 5))

    return np.tile(attmap, (Nx,1))

# %%

def NarrowMaps(pitch, cen, f_g, nel, c, dt, step, dz, min_d, max_d, factor, f0):
    """_summary_

    Args:
        pitch (_type_): _description_
        cen (_type_): _description_
        f_g (_type_): _description_
        nel (_type_): _description_
        c (_type_): _description_
        dt (_type_): _description_
        step (_type_): _description_
        dz (_type_): _description_
        min_d (_type_): _description_
        max_d (_type_): _description_
        factor (_type_): _description_
        f0 (_type_): _description_

    Returns:
        _type_: _description_
    """    
    geom = rit_g(f_g, cen[:,1], c)

    grid, indexes, Nx, Nz = gridcreate(pitch, nel, step, dz, min_d, max_d)

    A = Narrow_att_map(grid[:,2], f0, factor, Nx)

    H = np.zeros((nel, Nx, Nz), dtype = np.complex128)

    for i in range(nel):
        H1 = narrowMap(pitch, c, dt, geom, grid, Nx, Nz, i + 0.5, cen, f0)
        H2 = narrowMap(pitch, c, dt, geom, grid, Nx, Nz, -i-1+0.5, cen, f0)
        H[i, :, :] = H1+H2

    return H * A[np.newaxis, :, :], grid, [Nx, Nz]

# %%
pitch = 0.0025
kerf = 0.0005
elevation = 0.005
el = 100
step = 0.25
dz = 1e-4
min_depth = 0.008
max_depth = 0.02
Nx = 20
Ny = 100
geomf = 0.02
c = 1540
dt = 1e-8

cen = element_discr(pitch, kerf, elevation, Nx, Ny)
grid, idx, Nx, Nz = gridcreate(pitch, el, step, dz, min_depth, max_depth)
g = rit_g(geomf, cen[:,1], c)
# %%

Hprova = narrowMap(pitch, c, dt, g, grid, Nx, Nz, 0.5, cen, 4e6)
# %%
