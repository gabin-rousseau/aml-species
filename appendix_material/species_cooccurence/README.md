# Analysis of Re-location sites based on existing fauna 

## Overview

This code was used to cluster species in Florida, and use the most populous species in the corresponding cluster to create a heatmap to validate and/or filter the habitat based predictions made. It performs clustering using K-Means, and uses a Kernel Density Estimation plot to generate a heatmap. Yellow/Red areas on the heatmap indicate a higher incidence of fauna similar to the original habitat of the species in Florida.

## Libraries and Versions 

Python version 3.11.7

The libraries used in the code are
- NumPy version: 1.26.4
- Pandas version: 2.1.4
- Matplotlib version: 3.8.0
- Scikit-learn version: 1.2.2
- seaborn version: 0.12.2
  
## Data Dependencies 

The following files are required to recreate the given results: 
- species_train.npz
- {species}_AdaBoosted.csv (Saved as an output from Tree2.py)

## Code

This code is fully run in Python3 Jupyter Notebooks. The output from Tree2.py for a given species is required before attempting to run it. 
The heatmap can be made independently as well, and does not require predictions from a species. However, to recreate plots as in, the predictions file is necessary.

## Modifications for Other Species 

The followinf variables need to be modified to run the same analysis for different species

species_list= A list of species from the cluster the species of interest has the highest occurence in (output from the cell above)
pred= csv file of predictions of species of interest from Tree2.py



