# This directory contains data useful for the project.

- **full_test_pairs_clean.npz**: contains two numpy arrays; "test_ids_clean" are the species ids, "test_locs_clean" are the matching coordinates.

- **habitats_fl.csv**: up-to-date Florida training data, contains 6 columns; first is the data index from the original training dataset, 'Lat' for latitudes, 'Lon' for longitudes, 'Cls' for the climate annotations from Koeppen-Geiger, 'species_present' for the iNaturalist ids, 'Alt' for the altitudes! 2 values are missing their altitudes: 667 and 668 (with Python indexing). 
