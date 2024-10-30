#%% Import modules
import pandas as pd
#Import data
habitats=pd.read_csv('habitat_locs.csv', index_col=[0])

#%%
#Keep coordinates that are within florida boundaries
habitats_fl_temp = habitats[ (habitats['Lat'] >= 24.27) & (habitats['Lat'] <= 31.20)]
habitats_fl = habitats_fl_temp[ (habitats_fl_temp['Lon'] <= -80.02) & (habitats['Lon'] >= -87.38)]
#print(len(habitats))
#print(len(habitats_fl))
#print(habitats_fl)

#Write to file
sp_fl=list(set(habitats_fl['species_present']))
with open('habitats_fl.txt','w') as spfile:
	spfile.write('\n'.join([str(n) for n in sp_fl]))