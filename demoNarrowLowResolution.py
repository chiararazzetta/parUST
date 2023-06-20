# %%
from source.classBP import BeamPattern

# %%
####### Generate a test Narrow Beam Pattern with default values #######

NB = BeamPattern(step=0.5, Ndepth=200, NelImm=50)

# Compute maps
NB.MapsCompute()

# Option to save the maps 
# NB.SaveMaps('testData/LowNarrow.pkl')

# If maps are precomputed you can reload them
# NB.LoadMaps('testData/LowNarrow.pkl')

# %%
# If maps are loaded you want to change delays profile by defining a different focus or number
# of active elements or both
NB.DelaysSet()

NB.BPcompute()
NB.BPplot(0)

# %%
