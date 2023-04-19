# %%
import numpy as np
import scipy

# %%
def gridcreate(pitch, el, step, Nz, min_depth, max_depth):
    """Function for computing grid specifications

    Args:
        pitch (float):  element length for translations on the probe
        el (int): number of elements
        step (float): fractional part of the probe element for spacing the x coordinates
        Nz (int): number of points along depth
        min_depth (float): minimum depth
        max_depth (float): maximum depth

    Returns:
        numpy.array: coordinates o fthe grid points
        numpy.int: array containing index for the grid
        int: number of points along x axis
    """    
    coordx = np.arange(-el , el, step, dtype = np.float32)
    coordx = coordx * pitch
    coordz = np.linspace(min_depth, max_depth, Nz, dtype = np.float32)

    Nx = coordx.shape[0]

    taglia = Nz * Nx

    Z, X = np.meshgrid(coordz, coordx)
    idxZ, idxX = np.meshgrid(np.arange(0, Nz, dtype = np.int32), np.arange(0, Nx, dtype = np.int32))

    grid = np.concatenate((np.reshape(X, (taglia, 1)), np.zeros((taglia, 1), dtype=np.float32), np.reshape(Z, (taglia, 1))), 1)
    idx = np.concatenate((np.reshape(idxX, (taglia, 1)), np.reshape(idxZ, (taglia, 1))), 1)

    return grid, idx, Nx
# %%

def resp_probe(path, nt, pad, step):
    """Function for computing the probe impulse response

    Args:
        path (string): path to the .tx. file containing the measurements (time - measure in each line)
        nt (int): pre-established array size
        pad (int): number of zeros for padding
        step (float): time step to be used

    Returns:
        numpy.float: probe impulse response
        list: indexes of the significant frequences
    """    
    proberesponse = np.loadtxt(path)
    size = proberesponse.shape[0]
    dt = (proberesponse[1,0] - proberesponse[0,0])
    num = int((size * dt) / step) + 1

    rnew = scipy.signal.resample(proberesponse[:,1], num)
    mean = np.mean(rnew)
    rnew = rnew - mean
    rpad = np.pad(rnew, (pad, nt - rnew.shape[0]), 'constant')

    rfreq = scipy.fft.fft(rpad)[: (nt+pad) // 2]

    rnorm = np.abs(rfreq)
    rnorm = rnorm/np.max(rnorm)

    rdB = 20 * np.log10(rnorm)
    rdB[rdB < -40] = -40

    imax = np.argmax(rdB)
    iminsx = np.argmin(np.flipud(rdB[:imax]))
    imindx = np.argmin(rdB[imax:])

    rfreq = rfreq[imax - iminsx: imindx + imax]

    return rfreq, [imax - iminsx, imindx + imax]

# %%

def element_discr(pitch, kerf, elevation, Nx, Ny):
    """Function for calculating the grid of a single element for impulse response calculation

    Args:
        pitch (float):  element length for translations on the probe
        kerf (float):  space between elements (pitch = kerf + width)
        elevation (float): height of the element
        Nx (int): number of point along x-axis
        Ny (int):number of point along y-axis

    Returns:
        _type_: _description_
    """    
    width = pitch - kerf
    dx = width / Nx
    dy = elevation / Ny

    x = np.arange(- width / 2, width / 2, dx)
    y = np.arange(- elevation / 2, elevation / 2,dy)
    xc = x + dx/2
    yc = y + dy/2
    gridx, gridy = np.meshgrid(xc,yc)
    return np.array([np.reshape(gridx, (1, Nx * Ny)), np.reshape(gridy,(1, Nx * Ny)),np.zeros((1, Nx * Ny))], dtype=np.single)[:,0,:].T

# %% 

def rit_g(geomf, y_cen, c):
    """ Function for calculating the geometric delays (simulation of the lens curvature)

    Args:
        geomf (float): depth of the lens geometrical focus
        y_cen (numpy.float): array containing the coordinate of the elemet discretizaion
        c (float): medium speed

    Returns:
       numpy.float: array containing the geometric delays
    """    
    return (np.sqrt(y_cen ** 2 + geomf ** 2) / c) - (abs(geomf) / c)
# %%
