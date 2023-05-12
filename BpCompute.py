# %%
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt


# %%
def std_del(z_f, pitch, c, el, device = "cpu"):
    """Function for standard delay calculation

    Args:
        z_f (float): depth of focus
        pitch (float):  element length for translations on the probe
        c (float): medium speed of sound
        el (int): number of actuve elements
        device (string): flag to enable gpus

    Returns:
        array.float: array containing standard delays
    """
    if device == "cpu":
        return (z_f - np.sqrt(((0.5 + np.arange(0, el)) * pitch) ** 2 + z_f**2)) / c
    elif device == "gpu":
        return (z_f - cp.sqrt(((0.5 + cp.arange(0, el)) * pitch) ** 2 + z_f**2)) / c


def del_to_freq(delays, dt, ntimes, pad, nfreq=None, device = "cpu"):
    """Function to pass the delays in time frequency domain

    Args:
        delays (array.float): transmission delays
        dt (float): time step
        ntimes (int): maximum number of temporal instant
        pad (int): number of zerod for signals padding
        nfreq (list, optional): indexes of min and max the significant frequences. Defaults to None.
        device (string): flag to enable gpus

    Returns:
        array.complex: array for the delays in frequency
    """

    if device == "cpu":
        if nfreq is None:
            w = np.tile(np.linspace(0, 1 / dt, pad + ntimes + 1), (delays.shape[0], 1))
        else:
            w = np.tile(
                np.linspace(0, 1 / dt, pad + ntimes + 1)[nfreq[0] : nfreq[1]],
                (delays.shape[0], 1),
            )[:, : (ntimes + pad) // 2]

        w = w[:, : int((ntimes + pad) / 2)]

    
        return np.exp(-2 * np.pi * 1j * w * delays[:, np.newaxis])
    elif device == "gpu":
        if nfreq is None:
            w = cp.tile(cp.linspace(0, 1 / dt, pad + ntimes + 1), (delays.shape[0], 1))
        else:
            w = cp.tile(
                cp.linspace(0, 1 / dt, pad + ntimes + 1)[nfreq[0] : nfreq[1]],
                (delays.shape[0], 1),
            )[:, : (ntimes + pad) // 2]

        w = w[:, : int((ntimes + pad) / 2)]

    
        return cp.exp(-2 * cp.pi * 1j * w * delays[:, cp.newaxis])



# %%
def NarrowBP(delay, map, attenuation, f0, elements, device = "cpu"):
    """Function for Narrow Beam Pattern computation

    Args:
        delay (array.float): array containing the values of the delays
        map (array.complex): array of impulse response maps
        attenuation (array.complex): attenuation map
        f0 (float): trasmission frequency
        elements (int): number of active elements (half aperture)
        device (string): flag to enable gpus

    Returns:
        array.float: beam pattern power values
    """
    if device == "cpu":
        delayed = (
            map[:elements, :, :]
            * np.exp(-2 * np.pi * 1j * f0 * delay)[:elements, np.newaxis, np.newaxis]
        )
        B = np.sum(delayed, axis=0)
        return (np.abs(B * attenuation)) ** 2
    elif device == "gpu":
        map = cp.asarray(map)
        attenuation = cp.asarray(attenuation)
        
        delayed = (
            map[:elements, :, :]
            * cp.exp(-2 * cp.pi * 1j * f0 * delay)[:elements, cp.newaxis, cp.newaxis]
        )
        B = cp.sum(delayed, axis=0)
        return (cp.abs(B * attenuation)) ** 2


# %%
def wideMapCut(NelImm, step, H, Nz, grid, device = "cpu"):
    """Function to generate the multiple maps for Bp

    Args:
        NelImm (int): half-number of element to be visualized
        step (float): fractional part of the pitch for spacing the x coordinates
        H (array.complex): global wide map
        Nz (int): number of grid points along z axis
        grid (array.float): coordinates of the grid
        device (string): flag to enable gpus

    Returns:
        array.complex: set of maps for the desired elements
        int: the grid dimesions along x axis
        int: the grid dimesions along z axis
        array.float: coordinates of the image grid
    """
    n = int(1 / step)
    t = n * Nz

    if device == "cpu":
        centre = np.where(grid[:, 0] == 0)[0][0]
        N = int(NelImm / 2)

        Mapsize = [t * NelImm, H.shape[1]]
        Maps = np.empty((NelImm, Mapsize[0], Mapsize[1]), dtype=complex)
    elif device == "gpu":
        centre = cp.where(grid[:, 0] == 0)[0][0]
        N = int(NelImm / 2)

        Mapsize = [t * NelImm, H.shape[1]]
        Maps = cp.empty((NelImm, Mapsize[0], Mapsize[1]), dtype=complex)


    for i in range(-N, N):
        sx = centre - t * (N + i)
        dx = centre + t * (N - i)

        Maps[N + i, :, :] = H[sx:dx, :]

    return Maps, n * NelImm, Nz, grid[centre - t * N : centre + t * N + Nz, :]


# %%
def WideBP(delay, map, elements, dt, ntimes, pad, Nx, Nz, I, nfreq=None, device = "cpu"):
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
        device (string): flag to enable gpus

    Returns:
        array.float: beam pattern power values
    """

    if device == "cpu":
        e = np.concatenate((np.flipud(delay), delay))
        del_freq = del_to_freq(e, dt, ntimes, pad, nfreq, device)

        center = int(map.shape[0] / 2)
        delayed = np.sum(
            map[center - elements : center + elements, :, :] * del_freq[:, np.newaxis, :],
            axis=0,
        )

        pulsed = delayed * I[np.newaxis, :]

        power = np.sum(np.abs(pulsed) ** 2, axis=1)
        return np.reshape(power, (Nx, Nz))
    elif device == "gpu":
        map = cp.asarray(map)
        e = cp.concatenate((cp.flipud(delay), delay))
        del_freq = del_to_freq(e, dt, ntimes, pad, nfreq, device)

        center = int(map.shape[0] / 2)
        delayed = cp.sum(
            map[center - elements : center + elements, :, :] * del_freq[:, cp.newaxis, :],
            axis=0,
        )

        pulsed = delayed * I[cp.newaxis, :]

        power = cp.sum(cp.abs(pulsed) ** 2, axis=1)
        return cp.reshape(power, (Nx, Nz))


# %%
def todB(BP, cut=-40, device = "cpu"):
    """Function for computing the valuein deciBel of the Beam patterns

    Args:
        BP (array.float): beam pattern value
        cut (int): threshold on decibels for visualization
        device (string): flag to enable gpus

    Returns:
        array.float: beam pattern deciBel values
    """
    if device == "cpu":
        MaxC = np.max(np.max(BP))
        Cnorm = BP / MaxC
        Cnorm = 10 * np.log10(Cnorm)
    
    elif device == "gpu":
        MaxC = cp.max(cp.max(BP))
        Cnorm = BP / MaxC
        Cnorm = 10 * cp.log10(Cnorm)
        
    Cnorm[Cnorm < cut] = cut
    return Cnorm


# %%
