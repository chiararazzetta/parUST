# %%
from classBP import BeamPattern
from time import time
# %%
####### Generate a test Narrow Beam Pattern with default values #######

NB = BeamPattern()

# Compute and saving of maps
NB.MapsCompute()
NB.SaveMaps('testData/testNarrow.pkl')

# If maps are precomputed you can reload them
# NB.LoadMaps('testData/testNarrow.pkl')

# If maps are loaded you want to change delays profile by defining a different focus or number
# of active elements or both
NB.beam["active_el"] = 20
NB.beam["focus"] = 30e-3
NB.DelaysSet()

NB.BPcompute()
NB.BPplot()

# %%
####### Generate a test Wide Beam Pattern with default values #######

WB = BeamPattern(BPtype='Wide', Ncycles=3, factor=0.3)

# Compute and saving of maps
WB.MapsCompute()
WB.SaveMaps('testData/testWide.pkl')
#%%
# If maps are# precomputed you can reload them
WB.LoadMaps('testData/testWide.pkl')

# If maps are loaded you want to change delays profile by defining a different focus or number
# of active elements or both
WB.beam["active_el"] = 20
WB.beam["focus"] = 30e-3
WB.DelaysSet()

# Computation and display of BP
WB.BPcompute()
WB.BPplot()
