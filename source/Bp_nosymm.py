# %%
import numpy as np
import cupy as cp


# %%
def delays(f, pitch, c, el, device="cpu"):
    """Function for standard delay calculation

    Args:
        f (array.float): array containing (x_f, z_f) focus coordinates
        pitch (float):  element length for translations on the probe
        c (float): medium speed of sound
        el (array.int): array containing the number of active elements on left 
                        and right side of the central line footpoint 
        device (string): flag to enable gpus

    Returns:
        array.float: array containing standard delays
    """
    if device == "cpu":
        return (f[1] - np.sqrt(((0.5 + np.arange(-el[0], el[1])) * pitch - f[0]) ** 2 + f[1]**2)) / c
    elif device == "gpu":
        return (f[1] - cp.sqrt(((0.5 + cp.arange(-el[0], el[1])) * pitch- f[0]) ** 2 + f[1]**2)) / c



def del_to_freq(delays, dt, ntimes, pad, nfreq=None, device="cpu"):
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
            w = np.tile(np.linspace(0, 1 / dt, pad + ntimes + 1),
                        (delays.shape[0], 1))
        else:
            w = np.tile(
                np.linspace(0, 1 / dt, pad + ntimes + 1)[nfreq[0]: nfreq[1]],
                (delays.shape[0], 1),
            )[:, : (ntimes + pad) // 2]

        w = w[:, : int((ntimes + pad) / 2)]

        return np.exp(-2 * np.pi * 1j * w * delays[:, np.newaxis])
    elif device == "gpu":
        if nfreq is None:
            w = cp.tile(cp.linspace(0, 1 / dt, pad + ntimes + 1),
                        (delays.shape[0], 1))
        else:
            w = cp.tile(
                cp.linspace(0, 1 / dt, pad + ntimes + 1)[nfreq[0]: nfreq[1]],
                (delays.shape[0], 1),
            )[:, : (ntimes + pad) // 2]

        w = w[:, : int((ntimes + pad) / 2)]

        return cp.exp(-2 * cp.pi * 1j * w * delays[:, cp.newaxis])


# %%
def apodiz(el, sigma, type="gauss"):
    """Function for initializing apodization

    Args:
        el (array.int): array containing the number of active elements on left 
                        and right side of the central line footpoint
        sigma (float): parameter
        type (str, optional): Type of window: gauss, hamming, hanning. Defaults to "gauss".

    Returns:
        array.float: apodization weights
    """
    elements = np.max(np.abs(el))
    if type == "gauss":
        A = np.exp(-0.5*((np.arange(0, elements)+0.5)/(sigma*elements/2)) ** 2)
    elif type == "hanning":
        A = 0.5*(1-np.cos((2*np.pi*(np.arange(0, 2*elements)+0.5))/elements))[:elements]
    elif type == "hamming":
        A = sigma - (1 - sigma) * np.cos((2*np.pi*(np.arange(0, 2*elements)+0.5)/elements))[:elements]
    return np.concatenate((np.flipud(A[:-el[0]]), A[:el[1]]))
# %%

def N_BP(delay, H, step, Nz, NelImm, grid, f0, elements, apo=0, sigma=1.5, type="gauss", device="cpu"):
    """Function for Narrow Beam Pattern computation

    Args:
        delay (array.float): array containing the values of the delays
        H (array.complex): array of impulse response maps
        step
        Nz
        NelImm
        attenuation (array.complex): attenuation map
        f0 (float): trasmission frequency
        elements (array.int): array containing the number of active elements on left 
                        and right side of the central line footpoint 
        apo (int, optional): flag to enable apodization: 0 no, 1 yes. Defaults to 0
        sigma (float, optional): parameter of apodization windows. Defaults to 1.5
        type (string, optional): type of apodization window, Defaults to "gauss"
        device (string, optional): flag to enable gpus. Defaults to "cpu"

    Returns:
        array.float: beam pattern power values
    """
    n = int(1 / step)
    t = n * Nz
    N = int(NelImm / 2)
    centre = np.where(grid[:, 0] == 0)[0][0] - Nz

    if device == "cpu":
        Map = np.zeros((t * NelImm), dtype=complex)
        
        if apo == 0:
            for i in range(elements[0], elements[1]):
                sx = centre - t * (N + i)
                dx = centre + t * (N - i)
                Map = Map + (H[sx:dx] * np.exp(-2 * np.pi * 1j * f0 * delay[-elements[0]+i]))
        elif apo == 1:
            weights = apodiz(elements, sigma, type)
            for i in range(elements[0], elements[1]):
                sx = centre - t * (N + i)
                dx = centre + t * (N - i)
                Map = Map + H[sx:dx, :] * np.exp(-2 * np.pi * 1j * f0 * delay[-elements[0]+i]) * weights[-elements[0]+i]
        
        return np.reshape(np.abs(Map) ** 2, (n * NelImm, Nz))
    elif device == "gpu":
        H = cp.asarray(H)
        attenuation = cp.asarray(attenuation)

        Map = cp.zeros((t * NelImm, H.shape[1]), dtype=complex)
        
        if apo == 0:
            for i in range(elements[0], elements[1]):
                sx = centre - t * (N + i)
                dx = centre + t * (N - i)
                Map = Map + (H[sx:dx] * cp.exp(-2 * cp.pi * 1j * f0 * delay[-elements[0]+i]))
        
        elif apo == 1:
            weights = cp.asarray(apodiz(elements, sigma, type))
            weights = apodiz(elements, sigma, type)
            for i in range(elements[0], elements[1]):
                sx = centre - t * (N + i)
                dx = centre + t * (N - i)
                Map = Map + H[sx:dx, :] * np.exp(-2 * cp.pi * 1j * f0 * delay[-elements[0]+i, cp.newaxis]) * weights[-elements[0]+i, cp.newaxis]
        
        B = cp.sum(Map, axis=0)
        return cp.reshape(cp.abs(B) ** 2, (n * NelImm, Nz))

# %%
def W_BP(delay, H, elements, step, NelImm, grid, dt, ntimes, pad, Nz, I, nfreq=None, apo=0, sigma=1.5, type="gauss", device="cpu"):
    """Function for Wide Beam Pattern computation

    Args:
        delay (array.float): array containing the values of the delays
        H (array.complex): array of impulse response maps
        elements (int): number of active elements (half aperture)
        step (float): fractional part of the pitch for spacing the x coordinates
        NelImm (int): number of elements for depicting the BP
        grid (array.float): coordinates of the grid points
        dt (float): time step
        ntimes (int): maximum number of temporal instant
        pad (int): number of zerod for signals padding
        Nz (int): number of point along z-axis
        I (array.float): spectrum of the emitted waveform (real frequencies)
        nfreq (list, optional): indexes of min and max the significant frequences. Defaults to None.
        apo (int, optional): flag to enable apodization: 0 no, 1 yes. Defaults to 0
        sigma (float, optional): parameter of apodization windows. Defaults to 1.5
        type (string, optional): type of apodization window, Defaults to "gauss"
        device (string, optional): flag to enable gpus.

    Returns:
        array.float: beam pattern power values
        int: dimensions of the grid along x-axis
        int: dimension of the grid alog z-axis
        array.float:new grid coordinates
    """
    n = int(1 / step)
    t = n * Nz
    N = int(NelImm / 2)
    centre = np.where(grid[:, 0] == 0)[0][0] - Nz

    if device == "cpu":
        Map = np.zeros((t * NelImm, H.shape[1]), dtype=complex)

        e = np.concatenate((np.flipud(delay), delay))
        del_freq = del_to_freq(e, dt, ntimes, pad, nfreq, device)

        if apo == 0:
            for i in range(elements[0], elements[1]):
                sx = centre - t * (N + i)
                dx = centre + t * (N - i)

                if nfreq is None:
                    Map = Map + (H[sx:dx, :] * del_freq[elements[0]+i, :])
                else:
                    Map = Map + (H[sx:dx, nfreq[0]:nfreq[1]]
                                 * del_freq[elements[0]+i, :])
        elif apo == 1:
            weights = apodiz(elements, sigma, type)
            weights = np.concatenate((np.flipud(weights), weights))
            for i in range(elements[0], elements[1]):
                sx = centre - t * (N + i)
                dx = centre + t * (N - i)

                if nfreq is None:
                    Map = Map + (H[sx:dx, :] * del_freq[elements+i, :]
                                 * weights[elements[0]+i, np.newaxis])
                else:
                    Map = Map + (H[sx:dx, nfreq[0]:nfreq[1]] *
                                 del_freq[elements[0]+i, :] * weights[elements[0]+i, np.newaxis])

        pulsed = Map * I[np.newaxis, :]
        power = np.sum(np.abs(pulsed) ** 2, axis=1)

        return np.reshape(power, (n * NelImm, Nz)), n * NelImm, Nz, grid[centre - t * N: centre + t * N + Nz, :]

    elif device == "gpu":

        Map = cp.zeros((t * NelImm, H.shape[1]), dtype=complex)

        H = cp.asarray(H)
        e = cp.concatenate((cp.flipud(delay), delay))
        del_freq = del_to_freq(e, dt, ntimes, pad, nfreq, device)

        if apo == 0:
            for i in range(elements[0], elements[1]):
                sx = centre - t * (N + i)
                dx = centre + t * (N - i)

                if nfreq is None:
                    Map = Map + (H[sx:dx, :] * del_freq[elements[0]+i, :])
                else:
                    Map = Map + (H[sx:dx, nfreq[0]:nfreq[1]]
                                 * del_freq[elements[0]+i, :])
        elif apo == 1:
            weights = cp.asarray(apodiz(elements, sigma, type))
            for i in range(elements[0], elements[1]):
                sx = centre - t * (N + i)
                dx = centre + t * (N - i)

                if nfreq is None:
                    Map = Map + (H[sx:dx, :] * del_freq[elements[0]+i, :]
                                 * weights[elements[0]+i, cp.newaxis])
                else:
                    Map = Map + (H[sx:dx, nfreq[0]:nfreq[1]] *
                                 del_freq[elements[0]+i, :] * weights[elements[0]+i, cp.newaxis])

        pulsed = Map * I[cp.newaxis, :]
        power = cp.sum(cp.abs(pulsed) ** 2, axis=1)

        return cp.reshape(power, (n * NelImm, Nz)), n * NelImm, Nz, grid[centre - t * N: centre + t * N + Nz, :]

