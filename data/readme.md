# This directory contains data useful for the project.

## Files
- NEW **train_USA_FLspecies_full.zip** : contains a .csv with the same name: this is the data combining the two files below. The 'complete' training data, for when we want to put absences together with the original presence data.

- NEW **train_USA_FLspecies_pseudoabs.zip** : contains a .csv with the same name: same data as below, but pseudoabsences! (all 0's in the 'presence' column). This was made by using my "coordinate box difference" method to get something like target-group background points, which tend to be preffered for pseudoabsences by having them share the sampling bias of the actual presence data (because they are technically presence points for another species, if my understanding is correct). See https://doi.org/10.1016/j.ecoinf.2024.102623 and  https://doi.org/10.1890/07-2153.1 for more info.

- NEW **train_USA_FLspecies.csv** : This contains all (train) presence data points for Florida species throughout the US. (as of 10/11/2024) All elevation levels I defined (see below) are represented in the data (column is 'Alt_zone'). Added the column 'presence'. The only value you'll find in that column is 1, for now. That is because I have yet to generate pseudo-absences and annotate them.

- NEW **test_USA.csv** : features all test coordinates from the US with climate and elevation annotations. There is no species information in this dataset because every coordinate in here may or may not be assigned to a species, or multiple species. However, it is a good reference for that reason. However, you might prefer using the data below:

- NEW **test_USA_full.csv** : every test species presence datapoint, with climate and elevation annotations! Basically the same columns as train_USA_FLspecies.csv. If I manage to introduce absences in the training data, I should probably also specify test absences in this file.

- **full_test_pairs_clean.npz** : contains two numpy arrays; "test_ids_clean" are the species ids, "test_locs_clean" are the matching coordinates. This is the fully unwrapped test data, clean of unassigned coordinates.

- **habitats_fl.csv** : up-to-date Florida training data, contains 6 columns; first is the data index from the original training dataset, 'Lat' for latitudes, 'Lon' for longitudes, 'Cls' for the climate annotations from Koeppen-Geiger, 'species_present' for the iNaturalist ids, 'Alt' for the altitudes! 2 values are missing their altitudes: 667 and 668 (with Python indexing). 

## Details of the ordinal levels used to define elevation zones:
0. **Below 0:** sub-sea level

1. **Between 0 and 25:** coastal lowlands and plains (Chapter 21-Ecological subregions of the United 
States, capped by the max altitude in Western Florida coastal lowlands)

2. **Between 25 and 100:** lowlands (and the rest of the levels: mainly inspired by chapter 33 and the book "The Biology of Alpine Habitats")

3. **Between 100 and 500:** foothill zones 

4. **Between 500 and 1000:** uphill zones 

5. **Between 1000 and 1500:** lower montane 

6. **Between 1500 and 2000:** montane 

7. **Between 2000 and 2500:** upper montane 

8. **Between 2500 and 3000:** lower alpine 

9. **3000 and above:** alpine

## The "coordinate box difference" method
![Doodle of what my code did to select points for absence annotations](https://github.com/gabin-rousseau/aml-species/blob/main/figures/gabin/1111_boxdifference.png)
