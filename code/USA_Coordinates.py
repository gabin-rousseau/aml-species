#%% import
import numpy as np
import pandas as pd
import reverse_geocode

#%% loading test data 
data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']    # 2D array, rows are number of datapoints 
                                      # and columns are "latitude" and "longitude"
# data_test['test_pos_inds'] is a list of lists, where each list corresponds to 
# the indices in test_locs where a given species is present, it can be assumed 
# that they are not present in the other locations 
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))

#%% USA coordinates
country_dict = reverse_geocode.search(test_locs)
country_code = [d.get('country_code') for d in country_dict]
states = [d.get('state') for d in country_dict]

USA_coords=pd.DataFrame(columns=['Lat', 'Lon'])
USA_coords['Lat'] = [test_locs[l][0] for l in range(0,len(test_locs))]
USA_coords['Lon'] = [test_locs[l][1] for l in range(0,len(test_locs))]
USA_coords['country_code'] = country_code
USA_coords['state'] = states

USA_coords = USA_coords[USA_coords['country_code'] == 'US']
print(USA_coords.tail())
print(len(USA_coords))
