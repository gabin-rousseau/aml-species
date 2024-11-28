# Analysis of Re-location sites based on existing fauna 
## Overview
This code was used to cluster species in Florida, and use the most populous species in each cluster to create a heatmap to validate and/or filter the habitat based predictions made. It performs clustering using K-Means, and uses a Kernel Density Estimation plot to generate a heatmap. Yellow/Red areas on the heatmap indicate a higher incidence of fauna similar to the original habitat of the species in Florida.

##Libraries and Versions 

Python version 3.11.7

The libraries used in the code are
- NumPy version: 1.26.4
- Pandas version: 2.1.4
- Matplotlib version: 3.8.0
- Scikit-learn version: 1.2.2
- seaborn version: 0.12.2
  
##Data Dependencies 

The following files are required to recreate the given results: 
- species_train.npz
- {species}_predictions.csv ((WRITE IN PIPS PREDICTION FILE AND POINT TO THE CODE THAT GENERATES IT HERE)

##Code

This code is fully run in Python3 Jupyter Notebooks. The output from Tree2.py for a given species is required before attempting to run it. 
