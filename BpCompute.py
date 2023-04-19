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
        numpy.complex: beam pattern power values
    """    
    delayed = map[:elements, :, :] * np.exp(-2 * np.pi * 1j * f0 * delay)[:elements, np.newaxis, np.newaxis]
    B = np.sum(delayed, axis=0)
    return (np.abs(B * attenuation)) ** 2

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
