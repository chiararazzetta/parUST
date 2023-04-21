# %%
from MapGeneration import WideMaps
from f_initialization import element_discr, sinusoidalPulse
from BpCompute import std_del, wideMapCut, WideBP, todB
import pickle
import matplotlib.pyplot as plt

# %%

# Wide Maps Generation example
pitch = 0.245e-3
kerf = 0.035e-3
elevation = 5e-3
el = 70
step = 0.25
min_depth = 0.002
max_depth = 0.042
Nx = 40
Ny = 100
Nz = 400
geomf = 0.025
c = 1540
dt = 1e-8
ntimes = 1800
pad = 248
factor = 0.3

cen = element_discr(pitch, kerf, elevation, Nx, Ny)

H, grid_dim, grid, index = WideMaps(pitch, cen, geomf, el, c, dt, step, Nz, min_depth, max_depth, factor, ntimes, pad, None)

fWide = open('testData/testWide.pkl', 'wb')
pickle.dump([H, grid_dim, grid, index], fWide)
fWide.close()
# %%

#### If you have precomputed maps:
# fWide = open('testData/testWide.pkl', 'rb')
# H, grid_dim, grid, index = pickle.load(fWide)
# fWide.close()

# Wide BP computation Example
focus = 0.03
active_el = 20
f0 = 3e5
Ncycles = 3
Nimm = 50

rit = std_del(focus, pitch, c, active_el)
I = sinusoidalPulse(f0, Ncycles, dt, ntimes, pad, None)

maps, dim, grid_new = wideMapCut(Nimm, step, H, Nz, grid)

BP = WideBP(rit, maps, active_el, dt, ntimes, pad, dim[0], dim[1], I, None)

BPdeci = todB(BP)

plt.imshow(BPdeci, cmap = 'jet')
# %%
