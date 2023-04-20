# %%
from MapGeneration import NarrowMaps
from f_initialization import element_discr
from BpCompute import std_del, NarrowBP, todB
import pickle
import matplotlib.pyplot as plt

# %%

# Narrow Maps Generation example
pitch = 0.245e-3
kerf = 0.035e-3
elevation = 5e-3
el = 96
step = 0.25
min_depth = 0.002
max_depth = 0.042
Nx = 40
Ny = 100
Nz = 400
geomf = 0.025
c = 1540
dt = 1e-8
f0 = 4e6
factor = 0.5

cen = element_discr(pitch, kerf, elevation, Nx, Ny)

H, A, grid, grid_dim = NarrowMaps(pitch, cen, geomf, el, c, dt, step, Nz, min_depth, max_depth, factor, f0)

fNarrow = open('Maps/testNarrow.pkl', 'wb')
pickle.dump([H, A, grid, grid_dim], fNarrow)
fNarrow.close()
# %%

# Narrow BP computation Example
focus = 0.025
active_el = 20

rit = std_del(focus, pitch, c, active_el)

BP = NarrowBP(rit, H, A, f0, active_el)

BPdeci = todB(BP)

plt.imshow(BPdeci, cmap = 'jet')

