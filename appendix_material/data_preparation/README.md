# This directory contains the appendix files for the data preparation section of AML group 21. (2024/25)

### PREPARED DATA
- train_USA_FLspecies_full.zip : Archive containing a .csv file with the same name. Training dataset derived from the iNaturalist data, paired with Köppen-Geiger climate annotations, elevation (via open-elevation API), elevation zone (arbitrarily defined), and country/state. Restricted to Floridian species in the USA. The original presence data has been complemented with pseudo-absences defined with the box-difference approach explained in the report.

- predictable_coordinates_USA.csv : Prediction dataset, derived from the IUCN data. Features U.S. coordinates with climate and elevation annotations.

- test_USA_fullv2.zip : Archive contining a .csv file with the same name. Test dataset featuring all U.S. species within the U.S. Presences/Absences distinguished, complemented by climate and elevation annotations.

### EXTERNAL DATA REQUIRED

- Koeppen-Geiger-ASCII.txt : Köppen-Geiger annotations associated with coordinates covering the Earth. Can also be downloaded from http://koeppen-geiger.vu-wien.ac.at/present.htm

### CODE (Python3 Jupyter notebooks, please run after starting a Jupyter server.)
_Warning_: These notebooks were were mainly written with the intention to have a series of functional coding blocks that can be manually adapted for the data when applying a task as needed (such as getting closest climate annotations). As such, some user tinkering would be required to fully replicate the data preparation process. In other words, the data preparation code is better visualised as a function library rather than streamlined code that directly enables full replication of the data preparation process. 

- (0) unwrap_testdata.ipynb : early code to get all coordinate-species pairs from the IUCN test data.

- (1) prepare_climate_data.ipynb : Jupyter notebook used to prepare climate annotations.

- (2) US_data.ipynb : Jupyter notebook used for geographical filtering (isolate Floridian species in the training data, and U.S. coordinates in general).

- (3a) prepare_train_fl_elevation.ipynb : Jupyter notebook used to fetch and adapt elevation data in the training dataset.

- (3b) prepare_USA_elevation.ipynb : Jupyter notebook used to fetch and adapt elevation data in the prediction dataset. (Used for corresponding coordinates in the test dataset as well.)

- (4) prepare_pseudo-absences.ipynb : Jupyter notebook to establish presences/absences in the data, as well as final adjustements.
