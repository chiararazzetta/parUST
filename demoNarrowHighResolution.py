# %%
from source.classBP import BeamPattern

# %%
####### Generate a test Narrow Beam Pattern with default values #######

NB = BeamPattern()

# Compute maps
NB.MapsCompute()

# Option to save the maps 
# NB.SaveMaps('testData/testNarrow.pkl')

# If maps are precomputed you can reload them
# NB.LoadMaps('testData/Narrow45.pkl')

# %%
# If maps are loaded you want to change delays profile by defining a different focus or number
# of active elements or both
NB.beam["active_el"] = [15, 20]
NB.beam["focus"] = [10e-3, 30e-3]
NB.DelaysSet()

NB.BPcompute()
NB.BPplot(0)
# %%
