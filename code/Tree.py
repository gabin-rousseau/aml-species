#%% Importing modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#%% Import datasets
habitats=pd.read_csv('habitat_locs.csv', index_col=[0]) #Just to test code, not used in tree!
df_train=pd.read_csv('train_USA_FLspecies_full.zip')
df_test=pd.read_csv('test_USA_full.csv', index_col = [0])

#%% One-hot encoding function
from sklearn.preprocessing import OneHotEncoder

def OneHot(df):
    reshape_Cls = pd.DataFrame(df['Cls'], columns = ['Cls'])
    onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)  # 'first' to drop the first category to avoid multicollinearity
    onehot_encoded = onehot_encoder.fit_transform(reshape_Cls) # Fit and transform the data
    onehot_encoded_Cls = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['Cls']))
    joined_1H = df.merge(onehot_encoded_Cls,left_index=True, right_index=True)
    return joined_1H

## Test with habitats dataset
print("Original Class")
print(habitats['Cls'])
print("\nOne-Hot Encoded Class")
print(OneHot(habitats))

#%% Define Train Dataset
#Observations and pseudoabsences of target species in USA
#Filter to target species
df_train_41301 = df_train[df_train['species']==41301].reset_index(drop=True)
df_train_41301_1H = OneHot(df_train_41301) #Onehot encoded classes

#Reorder and make alt zone numeric
df_train_41301_1H['Alt_zone'] = df_train_41301_1H['Alt_zone'].apply(pd.to_numeric, errors = 'coerce')
#Add unseen Cls to dataset
df_train_41301_1H[['Cls_Dwb', 'Cls_Dwc', 'Cls_ET', 'Cls_Cfc']] = np.zeros(shape=[len(df_train_41301_1H),4])
#print(df_train_41301_1H)

#Define Test dataset
#Observations of target species in USA, all other species presence set to 'absence'
df_test_1H = OneHot(df_test)
df_test_1H_abs = df_test_1H.copy()
df_test_1H_abs.loc[df_test_1H_abs['species'] != 41301,'presence'] = 0 #Set absence data

#Reordering and make alt zone numeric
Cls_Cfc = df_test_1H_abs.pop('Cls_Cfc')
df_test_1H_abs['Cls_Cfc'] = Cls_Cfc
df_test_1H_abs['Alt_zone'] = df_test_1H_abs['Alt_zone'].apply(pd.to_numeric, errors = 'coerce')

#%% New Decision Tree

#Train on target species global observations
X_train = df_train_41301_1H[(np.delete(df_train_41301_1H.columns[7:], 1))] #Features are: Alt zone and onehot classes
#Target variable is target species present in USA observations
y_train = list(map(str, df_train_41301_1H['presence'])) #Target variable

X_test = df_test_1H_abs[(np.delete(df_test_1H_abs.columns[6:], 1))] #Features are: Alt zone and onehot classes
y_test = list(map(str, df_test_1H_abs['presence']))

clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#%%Visualise tree
column_names = X_train.columns
with open("tree.dot", 'w') as f:
    f = export_graphviz(clf, out_file=f,
                        feature_names=column_names,  
                        class_names=list(set(y_train)),  
                        filled=True, rounded=True,  
                        special_characters=False)
    
# %% ### Visualise results ###

import geopandas as gpd
import descartes
from shapely.geometry import Point

tree_pred = df_test_1H_abs.copy()
tree_pred['y_pred'] = y_pred
pred_loc = tree_pred[tree_pred['y_pred'] == '1']
pred_loc_new = pred_loc[pred_loc['presence'] == 0]

#plot
# plot train and test data for a random species
plt.close('all')
plt.figure(0)

# get test locations and plot
# test_inds_pos is the locations where the selected species is predicted to be present
# test_inds_neg is the locations where the selected species is predicted to be absent
test_inds_pos = np.where(pred_loc['presence'] == 0)[0]
test_inds_neg = np.where(pred_loc['presence'] == 1)[0]
plt.plot(pred_loc.iloc[test_inds_pos, 1], pred_loc.iloc[test_inds_pos, 0], 'b.', label='prediction')

# get train locations and plot
train_inds_pos = np.where(df_train_41301_1H['presence'] == 1)[0]
train_inds_neg = np.where(df_train_41301_1H['presence'] == 0)[0]
plt.plot(df_train_41301_1H.iloc[train_inds_pos, 1], df_train_41301_1H.iloc[train_inds_pos, 0], 'rx', label='train')

plt.title('41301')
plt.grid(True)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.ylabel('latitude')
plt.xlabel('longitude')
plt.legend()
plt.show()