#### Using K-nearest neightbour to cluster coordinates
#%%Import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


#%%Load data
D_train = np.load('species_train.npz')
train_locs = D_train['train_locs']  # 2D array, rows are number of datapoints and 
                                 # columns are "latitude" and "longitude"
train_ids = D_train['train_ids']    # 1D array, entries are the ID of the species 
                                 # that is present at the corresponding location in train_locs
species = D_train['taxon_ids']      # list of species IDe. Note these do not necessarily start at 0 (or 1)
species_names = dict(zip(D_train['taxon_ids'], D_train['taxon_names']))  # latin names of species 

#%% Cluster Coordinates
#Get habitat data
koeppen=pd.read_csv('Koeppen-Geiger-ASCII.txt',
                      delim_whitespace=True)
habitats=pd.DataFrame(train_locs,
                          columns=['Lat', 'Lon'],
                          dtype='float')

# Make coordinates into a list
coordinates=list(zip(koeppen['Lat'], koeppen['Lon']))

#Fit k-nearest neighbour
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(coordinates, koeppen['Cls'])

#Map predictions to training data coordinates
coords_train = list(zip(habitats['Lat'],habitats['Lon']))
habitats['Cls']=knn.predict(coords_train)



#%% Testing and reformatting
habitats['species_present']=train_ids
print(habitats.head())
print(habitats.tail())
habitats.to_csv("habitat_locs")
