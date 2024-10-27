#%% Import modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


#%% Attempt to identify if a species is likely to be present given a location
#Example species using 42722 - Petaurus norfolcensis (Squirrel glider)
#Read data
habitats=pd.read_csv('habitat_locs.csv', index_col=[0])
#print(habitats.head())
#print(habitats.tail())

squi_locs=habitats.copy()
squi_locs.loc[squi_locs['species_present'] == 42722, 
              'species_present'] = 1
squi_locs.loc[squi_locs['species_present'] != 1, 
              'species_present'] = 0

print(squi_locs.sample(10))
print(squi_locs.loc[squi_locs['species_present'] == 1])

#%% Decision Tree
#Adapted from https://www.datacamp.com/tutorial/decision-tree-classification-python?dc_referrer=https%3A%2F%2Fwww.google.com%2F
##Import modules
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#%%Super simple decision tree
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
X = squi_locs[['Lat', 'Lon']] #Features
y = squi_locs['species_present'] #Target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1) # 80% training and 20% test


# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Over-represented 0 sample - not a great classifier?