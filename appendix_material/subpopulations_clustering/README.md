# README: Clustering of species subpopulations in Florida
## Code Function Overview 
This code was used to cluster species from the dataset occurring in Florida. It performs clustering using 3 algorithms: K-means, K-medoids and agglomerative clustering from scikit-learn 1.5.2 and computes silhouette scores for 2-10 clusters to choose algorithms with the most well-defined clusters and optimal K. This approach can be applied to one or multiple species by adding IDs of species of interest to the `florida_specie_ids` list. 

## Technologies Used
Python 3.12.3 

## Pre-requisites 
This code uses the following libraries, make sure you have the latest versions:
- **NumPy (`numpy`)** 
- **Matplotlib (`matplotlib.pyplot`)**
- **Pandas (`pandas`)** 
- **Scikit-learn (`sklearn`)**
- **Scikit-learn-extra (`sklearn_extra`)**
- **Seaborn (`seaborn`)**
  
### Required Imports:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
```

## Usage:
1. Open the code in Jupiter Notebook
2. Run the cells one by one
3. To analyse other species you can modify the `florida_specie_ids` list by changing species IDs.


