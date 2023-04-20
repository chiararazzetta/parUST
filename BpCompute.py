# %% 
import numpy as np

# %%
def std_del(z_f, pitch, c, el):
    """Function for standard delay calculation

    Args:
        z_f (float): depth of focus
        pitch (float):  element length for translations on the probe
        c (float): medium speed of sound
        el (int): number of actuve elements

    Returns:
        numpy.float: array containing standard delays
    """    
    return (z_f - np.sqrt(((0.5 + np.arange(0, el)) * pitch) ** 2 + z_f ** 2))/c

def del_to_freq(delays, dt, ntimes, pad, nfreq = None):
    """Function to pass the delays in time frequency domain

    Args:
        delays (array.float): transmission delays
        dt (float): time step
        ntimes (int): maximum number of temporal instant
        pad (int): number of zerod for signals padding
        nfreq (list, optional): indexes of min and max the significant frequences. Defaults to None.

    Returns:
        array.complex: array for the delays in frequency
    """    
    if nfreq is None:
        w = np.tile(np.linspace(0, 1 / dt, pad + ntimes + 1), (delays.shape[0], 1))
    else:
        w = np.tile(np.linspace(0, 1 / dt, pad + ntimes + 1)[nfreq[0] : nfreq[1]], (delays.shape[0], 1))

    return np.exp(-2 * np.pi * 1j * w * delays[:, np.newaxis])
# %%
def NarrowBP(delay, map, attenuation, f0, elements):
    """Function for Narrow Beam Pattern computation

    Args:
        delay (array.float): array containing the values of the delays
        map (array.complex): array of impulse response maps
        attenuation (array.complex): attenuation map
        f0 (float): trasmission frequency
        elements (int): number of active elements (half aperture)

    Returns:
        numpy.float: beam pattern power values
    """    
    delayed = map[:elements, :, :] * np.exp(-2 * np.pi * 1j * f0 * delay)[:elements, np.newaxis, np.newaxis]
    B = np.sum(delayed, axis=0)
    return (np.abs(B * attenuation)) ** 2

# %%
def wideMapCut(NelImm, step, H, Nz, Nx, grid):
    """Function to generate the multiple maps for Bp

    Args:
        NelImm (int): half-number of element to be visualized
        step (float): fractional part of the pitch for spacing the x coordinates
        H (array.complex): global wide map
        Nz (int): number of grid points along z axis
        Nx (int): number of grid points along x axis
        grid (array.float): coordinates of the grid

    Returns:
        array.complex: set of maps for the desired elements
        list: dimensions of the grid
        array.float: coordinates of the image grid
    """    
    n = int(1 / step)
    t = n * Nz
    c = np.where(grid[:,0] == 0)[0][0]
    N = int(NelImm / 2)

    Mapsize = [t * NelImm, H.shape[1]]
    Maps = np.empty((NelImm, Mapsize[0], Mapsize[1]), dtype=np.complex)

    for i in range(-N, N):
        sx = c - t * (N + i)
        dx = c + t * (N - i)

        Maps[N + i, :, :] = H[sx: dx, :]

    return Maps, [Nx, Nz], grid[c - t * N: c + t * N + Nz, :]

# %% 
def WideBP(delay, map, elements, dt, ntimes, pad, Nx, Nz, I, nfreq = None):
    """Function for generating Wide Band Beam Patterns

    Args:
        delay (array.float): array containing the values of the delays
        map (array.complex): array of impulse response maps
        elements (int): number of active elements (half aperture)
        dt (float): time step
        ntimes (int): maximum number of temporal instant
        pad (int): number of zerod for signals padding
        Nz (int): number of point along the zaxis
        Nx (int): number of point along the xaxis
        I (array.complex): temporal frequency domain pulse
        list (optional): indexes of min and max the significant frequences. Defaults to None.

    Returns:
        array.float: beam pattern power values
    """    
    e = np.concatenate(np.flipud(delay), delay)
    del_freq = del_to_freq(e, dt, ntimes, pad, nfreq)

    delayed = np.sum(map[-elements:elements, :, :] * del_freq[:, np.newaxis, :], axis = 0)
    pulsed = delayed * I[np.newaxis, :]
    power = np.abs(np.sum(pulsed, axis = 1)) ** 2
    return np.reshape(power, (Nx, Nz))

# %%
def todB(BP):
    """Function for computing the valuein deciBel of the Beam patterns

    Args:
        BP (numpy.float): beam pattern value

    Returns:
        numpy.float: beam pattern deciBel values
    """    
    MaxC = np.max(np.max(BP))
    Cnorm = BP/MaxC
    Cnorm = 10 * np.log10(Cnorm)
    Cnorm[Cnorm < -40] = -40
    return Cnorm


# %%
