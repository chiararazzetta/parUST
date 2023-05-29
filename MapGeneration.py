# %%
import ctypes
from numpy.ctypeslib import load_library
from numpy.ctypeslib import ndpointer
import math
import numpy as np
import scipy 

from f_initialization import rit_g, grid_create, grid_create, resp_probe

# %%
risplib = load_library("matrix.so", ".")

H = risplib.hmat

H.restype = ndpointer(ctypes.c_float)
H.argtypes = [
    ndpointer(ctypes.c_float),
    ctypes.c_int,
    ndpointer(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_int,
    ndpointer(ctypes.c_float),
    ctypes.c_int,
    ndpointer(ctypes.c_float),
    ndpointer(ctypes.c_float),
]


def mrisptopy(cen, p, v, dt, g, ntimes):
    """Function to link python objects to c function

    Args:
        cen (array.float): array containing the coordinates of element discretization points
        p (array.float): array containing the coordinates of the points of the field
        v (float):medium speed
        dt (float): time step
        g (array.float): array containing the geometric delays to simulate the probe lens

    Returns:
        list : two array arrays, the array of impulse response values and the corresponding time
    """
    row = cen.shape[0]
    npunti = p.shape[0]
    h = np.zeros((npunti, ntimes), dtype=np.float32)
    t = np.zeros(npunti, dtype=np.float32)

    H(cen, row, p, npunti, v, dt, ntimes, g, 1, h, t)

    return h / row, t


# %%


def wideMap(c, dt, geom, grid, cen, nt, pad, A, hprobe=None):
    """Function to generate a impulse response map for a grid of points and a fixed probe element
        in Wide Band pulse transmission setting

    Args:
        c (float): medium speed
        dt (float): time step
        geom (array.float): array containing the geometric delays to simulate the probe lens
        grid (array.float): array containing the coordinates of the points of the field
        cen (array.float): array containing the coordinates of element discretization points
        nt (int): maximum number of temporal instant
        pad (int): number of zero for signals padding
        A (array.float): array containing the attenuation coefficients

    Optional Args:
        hprobe (list): if it is possible to have a measurements of the probe impulse response, we cut the significant frequencies
                       and take in account its shape

    Returns:
        array.complex: array containing the collection of the impulse responses
    """

    h, t = mrisptopy(cen, grid, c, dt, geom, nt)
    h = np.pad(h, ((0, 0), (int(pad/2), int(pad/2))))
    scipy.fft.set_global_backend("scipy")
    
    h = scipy.fft.fft(h, axis=1)
    h = h[:, : (nt + pad) // 2]

    trif = t - grid[:, 2] / c

    freq = np.linspace(0, 1 / dt, pad + nt + 1)[: (nt + pad) // 2]
    freq = np.repeat(freq.reshape([1, (nt + pad) // 2]), trif.shape, axis=0)

    if hprobe is None:
        phase = np.exp(-2 * math.pi * 1j * freq * trif[:, np.newaxis])
        h = h * phase * A
    else:
        hp = hprobe[0]
        dim = hprobe[1]
        phase = np.exp(
            -2 * math.pi * 1j * freq[:, dim[0] : dim[1]] * trif[:, np.newaxis]
        )
        h = h[:, dim[0] : dim[1]] * phase * hp[np.newaxis, :] * A[:, dim[0] : dim[1]]

    return h


# %%
def Narrow_att_map(coordz, f0, factor, N):
    """Function for computing the attenuation map in single frequency Narrow Band case

    Args:
        coordz (array.float): array containing all the possible depths in the field
        f0 (float): central frequency
        factor (float): factor for attenuation rule (0.5 for narrow transmission 0.3 for wide transmission)
        N (int): number of point along the xaxis

    Returns:
        array.complex: array containing the attenuation map with grid size
    """

    attmap = 10 ** ((-factor / 20) * (f0 * coordz) / (10**4))

    return np.tile(attmap, (N, 1))


# %%


def Wide_att_map(coordz, Nx, Nz, factor, nt, pad, dt):
    """Function for computing the attenuation map in Wide Band case

    Args:
        coordz (array.float): array containing all the possible depths in the field
        Nx (int): number of point along the xaxis
        Nz (int): number of point along the zaxis
        factor (float): factor for attenuation rule
        nt (int): maximum number of temporal instant
        pad (int): number of zerod for signals padding
        dt (float): time step

    Returns:
        array.complex: array containing the attenuation map with grid size
    """
    w = np.linspace(0, 1 / dt, pad + nt + 1)
    wreal = w[0 : (nt + pad) // 2]

    attmap = np.empty([Nz, (nt + pad) // 2])

    for i in range(Nz):
        attmap[i, :] = 10 ** ((-factor / 20 ) * (coordz[i] * wreal) / (10**4))

    return np.tile(attmap, (Nx, 1))


# %%
#%%
def NarrowMaps(pitch, c, dt, f_g, step, Nz, min_d, max_d, factor, cen, f0, Nel, NelImm):
    """Function to generate a impulse response map for a grid of points and a fixed probe element
        in Narrow Band pulse transmission setting, with mono frequency approximation

    Args:
        pitch (float): element length for translations on the probe
        c (float): medium speed
        dt (float): time step
        f_g (float): geometrical focus
        step (float): fractional part of the probe element for spacing the x coordinates
        Nz (int): number of points along depth
        min_d (float): minimum depth
        max_d (float): maximum depth
        factor (float): attenuation factor
        cen (array.float): array containing the coordinates of element discretization points
        f0 (float): central frequency of the narrow pulse
        Nel (int): number of probe elements
        NelImm (int): number of image elements

    Returns:
        array.complex: the maps array
        array.complex: the attenuation array
        array.float: the grid coordinates
        int: the grid dimesions along x axis
        int: the grid dimesions along z axis
    """
    geom = rit_g(f_g, cen[:, 1], c)

    grid, indexes, Nx = grid_create(pitch, Nel, step, Nz, min_d, max_d)


    T = np.arange(0, 300 * dt, dt)
    phase = np.exp(-2 * math.pi * 1j * T * f0)

    h, t = mrisptopy(cen, grid, c, dt, geom, 300)

    trif = t - grid[:, 2] / c
    h = np.sum(h * phase, 1)
    H = h * np.exp(-2 * math.pi * 1j * f0 * trif)

    n = int(1 / step)
    t = n * Nz
    centre = np.where(grid[:, 0] == 0)[0][0]
    N = int(NelImm / 2)
    Nx = n * NelImm

    A = Narrow_att_map(grid[:Nz, 2], f0, factor, Nx)

    Maps = np.zeros((N, Nx, Nz), dtype=complex)

    for i in range(-N, N):
        sx = centre - t * (N + i)
        dx = centre + t * (N - i)

        if i < 0:
            Maps[-i - 1, :, :] = Maps[-i -1, :, :] + np.reshape(H[sx:dx], (Nx, Nz))
        else:
            Maps[i, :, :] = Maps[i, :, :] + np.reshape(H[sx:dx], (Nx, Nz))
    
    return Maps, A, grid[centre - t * N : centre + t * N + Nz, :], Nx, Nz

# %%
def WideMaps(
    pitch, cen, f_g, nel, c, dt, step, Nz, min_d, max_d, factor, ntimes, pad, path=None
):
    """Function to generate an overall map for the central element of the probe, taking in acount the symmetry of the field

    Args:
        pitch (float): element length for translations on the probe
        cen (array.float): array containing the coordinates of element discretization points
        f_g (float): geometrical focus of the probe
        nel (int): number of elements of the probe to be considered
        c (float): medium speed
        dt (float): time step
        step (float): fractional part of the probe element for spacing the x coordinates
        Nz (int): number of points along depth
        min_d (float): minimum depth
        max_d (float): maximum depth
        factor (float): attenuation factor
        ntimes (int): maximum number of temporal instant
        pad (int): number of zerod for signals padding
        path (string, optional):path to the .txt file containing the measurement of the probe impulse response. Defaults to None.

    Returns:
        array.complex: the attenuated map array
        int: the grid dimesions along x axis
        int: the grid dimesions along z axis
        array.float: the grid coordinates
        array.int: the grid indexes
        list: indexes of min and max the significant frequences

    """
    geom = rit_g(f_g, cen[:, 1], c)

    grid, indexes, Nx = grid_create(pitch, nel, step, Nz, min_d, max_d)

    A = Wide_att_map(grid[:Nz, 2], Nx, Nz, factor, ntimes, pad, dt)

    if path is None:
        n_freq = None
        H = wideMap(c, dt, geom, grid, cen, ntimes, pad, A, None)
    else:
        r_probe, n_freq = resp_probe(path, ntimes, pad, step)
        H = wideMap(c, dt, geom, grid, cen, ntimes, pad, A, [r_probe, n_freq])

    return H, Nx, Nz, grid, n_freq


# %%
