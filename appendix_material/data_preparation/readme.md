# This directory contains the appendix files for the data preparation section of AML group 21. (2024/25)

### DATA

- train_USA_FLspecies_full.zip : Archive containing a .csv file with the same name. Training dataset derived from the iNaturalist data, paired with Köppen-Geiger climate annotations, elevation (via open-elevation API), elevation zone (arbitrarily defined), and country/state. Restricted to Floridian species in the USA. The original presence data has been complemented with pseudo-absences defined with the box-difference approach explained in the report.

- predictable_coordinates_USA.csv : Prediction dataset, derived from the IUCN data. Features U.S. coordinates with climate and elevation annotations.

- test_USA_fullv2.zip : Archive contining a .csv file with the same name. Test dataset featuring all U.S. species within the U.S. Presences/Absences distinguished, complemented by climate and elevation annotations.


### CODE

- prepare_climate_data.ipynb : Jupyter notebook used to prepare climate annotations.

- US_data.ipynb : Jupyter notebook used for geographical filtering (isolate Floridian species in the training data, and U.S. coordinates in general).

- prepare_train_fl_elevation.ipynb : Jupyter notebook used to fetch and adapt elevation data in the training dataset.

- prepare_USA_elevation.ipynb : Jupyter notebook used to fetch and adapt elevation data in the prediction dataset. (Used for corresponding coordinates in the test dataset as well.)

- prepare_pseudo-absences.ipynb : Jupyter notebook to establish presences/absences in the data, as well as final adjustements.
