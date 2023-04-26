# %%
from classBP import BeamPattern

# %%
####### Generate a test Narrow Beam Pattern with defoult values #######

NB = BeamPattern()

# Compute and saving of maps
NB.MapsCompute()
NB.SaveMaps('testData/testNarrow.pkl')

# If maps are precomputed you can reload them
#NB.LoadMaps('testData/testNarrow.pkl')

# Computation and display of BP
NB.BPcalculate()
NB.BPplot()

# %%
####### Generate a test Wide Beam Pattern with defoult values #######

WB = BeamPattern(BPtype='Wide')

# Compute and saving of maps
WB.MapsCompute()
WB.SaveMaps('testData/testWide.pkl')

# If maps are precomputed you can reload them
#NB.LoadMaps('testData/testNarrow.pkl')

# Computation and display of BP
WB.BPcalculate()
WB.BPplot()