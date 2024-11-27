#%% IMPORT MODULES
# Importing modules
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import randint
from global_land_mask import globe
from pyinaturalist import get_taxa_by_id
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

#%% Import datasets
habitats=pd.read_csv('habitat_locs.csv', index_col=[0]) #Just to test code, not used in tree!
df_train_init=pd.read_csv('train_USA_FLspecies_full.zip')
df_test=pd.read_csv('test_USA_full.csv', index_col = [0])
df_test2=pd.read_csv('test_USA_fullv2.csv')
df_test3=pd.read_csv('predictable_coordinates_USA.csv')

#%% ONE HOT ENCODING FUNCTION
# One-hot encoding function

def OneHot(df):
    reshape_Cls = pd.DataFrame(df['Cls'], columns = ['Cls'])
    onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)  # 'first' to drop the first category to avoid multicollinearity
    onehot_encoded = onehot_encoder.fit_transform(reshape_Cls) # Fit and transform the data
    onehot_encoded_Cls = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['Cls']))
    joined_1H = df.merge(onehot_encoded_Cls,left_index=True, right_index=True)
    return joined_1H

#%% DEFINE CLASSES
classes = habitats['Cls'].unique().tolist()
#Define column names
cols = ['Lat', 'Lon', 'Cls', 'species', 'country_code', 'state', 'Alt',
       'Alt_zone', 'presence']
for c in classes:
    name = 'Cls_'+ c
    cols.append(name)

#%% DEFINE PREDICTION DATAFRAME
#Define prediction set
df_test3_1H = OneHot(df_test3)
df_test3_1H = df_test3_1H[globe.is_land(df_test3_1H['Lat'], df_test3_1H['Lon'])]
df_test3_1H = df_test3_1H.reindex(columns=cols, fill_value=0)
df_test3_1H['Alt_zone'] = df_test3_1H['Alt_zone'].apply(pd.to_numeric, errors = 'coerce')
df_test3_1H['presence'] = df_test3_1H['presence'].apply(pd.to_numeric, errors = 'coerce')


#%% # Define Test and Train Dataset
#Observations and pseudoabsences of target species in USA
#Filter to target species
def train_species(sp):
    df_train=pd.read_csv('train_USA_FLspecies_full.zip')
    df_train_sp = df_train[df_train['species']==int(sp)].reset_index(drop=True)
    df_train_sp = df_train_sp[globe.is_land(df_train_sp['Lat'], df_train_sp['Lon'])]
    df_train_1H = OneHot(df_train_sp) #Onehot encoded classes

    #Reorder and make alt zone numeric
    df_train_1H['Alt_zone'] = df_train_1H['Alt_zone'].apply(pd.to_numeric, errors = 'coerce')
    df_train_1H['presence'] = df_train_1H['presence'].apply(pd.to_numeric, errors = 'coerce')
    #Add unseen Cls to dataset
    df_train_1H = df_train_1H.reindex(columns = cols, fill_value=0)
    return df_train_1H
    
##Define Test dataset##
##Observations of target species in USA, all other species presence set to 'absence'

def test_species(sp):
    df_test2=pd.read_csv('test_USA_fullv2.csv')
    df_test2 = df_test2[globe.is_land(df_test2['Lat'], df_test2['Lon'])]
    df_test_1H = OneHot(df_test2)
    df_test_1H['presence'] = df_test_1H['presence'].apply(pd.to_numeric, errors = 'coerce')
    df_test_1H.loc[df_test_1H['species'] != int(sp), 'presence'] = 0 #Set absence data
    df_test_pos = df_test_1H.loc[df_test_1H['presence'] == 1]
    
    df_test_1H.drop_duplicates(subset=['Lat', 'Lon', 'presence'], inplace=True)


    df_test_1H['nearest_Lat'] = df_test_1H['Lat'].apply(lambda i: (i - df_test_pos['Lat']).abs().min())
    df_test_1H['nearest_Lon'] = df_test_1H['Lon'].apply(lambda i: (df_test_pos['Lon'] - i).abs().min())

    drop_indices = df_test_1H[(df_test_1H['presence'] == 0) &
                              (df_test_1H['nearest_Lat'] < 0.1) &
                              (df_test_1H['nearest_Lon'] < 0.1)].index
    if len(df_test_1H) - sum(df_test_1H['presence']) > 1000:
        df_test_1H_abs = df_test_1H.drop(index=drop_indices)
        #print('dropping species because df_test_1H is', len(df_test_1H['presence']), 'long')
    else:
        df_test_1H_abs = df_test_1H
    df_test_1H_abs = df_test_1H_abs.reindex(columns = cols, fill_value=0)

    return df_test_1H_abs

#%% BASIC TREE

from sklearn.tree import DecisionTreeClassifier, plot_tree

#Change species as needed
sp = 4146
sp_name = get_taxa_by_id(sp)["results"][0].get("name")

df_train = train_species(sp)[cols]
features = np.delete(df_train.columns[7:], 1)
tree_X_train = df_train[features]
tree_y_train = df_train['presence']

df_test = test_species(sp)[cols]
tree_X_test = df_test[features]
tree_X_test2 = df_test3_1H[features]
tree_y_test = df_test['presence']

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(tree_X_train, tree_y_train)


tree_y_pred = tree_clf.predict(tree_X_test).astype(int)
tree_y_pred2 = tree_clf.predict(tree_X_test2).astype(int)

# TREE SCORES
print("Accuracy:",metrics.accuracy_score(tree_y_test, tree_y_pred))
print("The precision score is: %.2f" % metrics.precision_score( tree_y_test, tree_y_pred, pos_label=1))
print("The recall score is: %.2f" % metrics.recall_score( tree_y_test, tree_y_pred, pos_label=1), "\n")
print("The F1 score is: %.2f" % metrics.f1_score( tree_y_test, tree_y_pred, pos_label=1))


cm = metrics.confusion_matrix(tree_y_test , tree_y_pred )
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=tree_clf.classes_)
disp.plot()

plt.figure(figsize=(20, 12))
plot_tree(tree_clf) 
plt.show()

tree_param_grid = {
    'max_depth': np.arange(2,30, 5),
    'class_weight': ['balanced'],
    'min_samples_leaf': np.arange(2, 10, 2),
    'criterion' : ['gini']
}

tree_grid_search = GridSearchCV(tree_clf, tree_param_grid, verbose=3, scoring= 'f1', cv=3)
tree_grid_search.fit(tree_X_train, tree_y_train)
tree_best_params = tree_grid_search.best_params_

tree_clf.set_params(**tree_best_params)
tree_clf.fit(tree_X_train, tree_y_train)

tree_y_pred = tree_clf.predict(tree_X_test).astype(int)
tree_y_pred2 = tree_clf.predict(tree_X_test2).astype(int)

# RANDOM TREE SCORES
print("Scores after tuning", tree_clf.get_params())
print("Accuracy:",metrics.accuracy_score(tree_y_test, tree_y_pred))
print("The precision score is: %.2f" % metrics.precision_score( tree_y_test, tree_y_pred, pos_label=1))
print("The recall score is: %.2f" % metrics.recall_score( tree_y_test, tree_y_pred, pos_label=1), "\n")
print("The F1 score is: %.2f" % metrics.f1_score( tree_y_test, tree_y_pred, pos_label=1))


cm = metrics.confusion_matrix(tree_y_test , tree_y_pred )
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=tree_clf.classes_)
disp.plot()

plt.figure(figsize=(20, 12))
plot_tree(tree_clf) 
plt.show()

#%% RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier # Import Random Tree Classifier
from sklearn.tree import export_graphviz

df_train = train_species(sp)[cols]
features = np.delete(df_train.columns[7:], 1)
tree_X_train = df_train[features]
tree_y_train = df_train['presence']

df_test = test_species(sp)[cols]
tree_X_test = df_test[features]
tree_X_test2 = df_test3_1H[features]
tree_y_test = df_test['presence']

tree_clf = RandomForestClassifier(random_state=42)
tree_clf.fit(tree_X_train, tree_y_train)


tree_y_pred = tree_clf.predict(tree_X_test).astype(int)
tree_y_pred2 = tree_clf.predict(tree_X_test2).astype(int)

# RANDOM TREE SCORES
print(sp_name, '(ID:' + str(sp) + ")\nMetrics before tuning: "+str(tree_clf.get_params()))
print("Accuracy:",metrics.accuracy_score(tree_y_test, tree_y_pred))
print("The precision score is: %.2f" % metrics.precision_score( tree_y_test, tree_y_pred, pos_label=1))
print("The recall score is: %.2f" % metrics.recall_score( tree_y_test, tree_y_pred, pos_label=1))
print("The F1 score is: %.2f" % metrics.f1_score( tree_y_test, tree_y_pred, pos_label=1))
print(tree_clf.get_params)

cm = metrics.confusion_matrix(tree_y_test , tree_y_pred )
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=tree_clf.classes_)
disp.plot()

tree_param_grid = {
    'max_depth': np.arange(2,30, 5),
    'class_weight': ['balanced'],
    'min_samples_leaf': np.arange(2, 10, 2),
    'criterion' : ['gini'],
}

tree_grid_search = GridSearchCV(tree_clf, tree_param_grid, verbose=3, scoring= 'f1', cv=3)
tree_grid_search.fit(tree_X_train, tree_y_train)
tree_best_params = tree_grid_search.best_params_

tree_clf.set_params(**tree_best_params)
tree_clf.fit(tree_X_train, tree_y_train)

tree_y_pred = tree_clf.predict(tree_X_test).astype(int)
tree_y_pred2 = tree_clf.predict(tree_X_test2).astype(int)

# RANDOM TREE SCORES
print(sp_name, '(ID:' + str(sp) + ")\nMetrics after tuning: "+str(tree_clf.get_params()))
print("Accuracy:",metrics.accuracy_score(tree_y_test, tree_y_pred))
print("The precision score is: %.2f" % metrics.precision_score( tree_y_test, tree_y_pred, pos_label=1))
print("The recall score is: %.2f" % metrics.recall_score( tree_y_test, tree_y_pred, pos_label=1))
print("The F1 score is: %.2f" % metrics.f1_score( tree_y_test, tree_y_pred, pos_label=1))

cm = metrics.confusion_matrix(tree_y_test , tree_y_pred )
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=tree_clf.classes_)
disp.plot()

#%% ADABOOST

from sklearn.ensemble import AdaBoostClassifier


df_train = train_species(sp)[cols]
ada_X_train = df_train[features]
ada_y_train = df_train['presence']

df_test = test_species(sp)[cols]
ada_X_test = df_test[features]
ada_X_test2 = df_test3_1H[features]
ada_y_test = df_test['presence']

ada_clf = AdaBoostClassifier(estimator = tree_clf, 
algorithm = 'SAMME',
random_state=42)
ada_clf.fit(ada_X_train, ada_y_train)


ada_y_pred = ada_clf.predict(ada_X_test).astype(int)
ada_y_pred2 = ada_clf.predict(ada_X_test2).astype(int)

# PRE_TUNING SCORES
print(sp_name, '(ID:' + str(sp) + ")\nMetrics before tuning: "+str(ada_clf.get_params()))
print("Accuracy:",metrics.accuracy_score(ada_y_test, ada_y_pred))
print("The precision score is: %.2f" % metrics.precision_score( ada_y_test, ada_y_pred, pos_label=1))
print("The recall score is: %.2f" % metrics.recall_score( ada_y_test, ada_y_pred, pos_label=1))
print("The F1 score is: %.2f" % metrics.f1_score( ada_y_test, ada_y_pred, pos_label=1))

cm = metrics.confusion_matrix(ada_y_test , ada_y_pred )
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=ada_clf.classes_)
disp.plot()

ada_param_grid = {
    'n_estimators': np.arange(50,225,25),
    'learning_rate': [0.01, 0.1, 1, 5, 10]
}

ada_grid_search = GridSearchCV(ada_clf, ada_param_grid, verbose=3, scoring= 'f1', cv=3)
ada_grid_search.fit(ada_X_train, ada_y_train)
ada_best_params = ada_grid_search.best_params_

ada_clf.set_params(**ada_best_params)
ada_clf.fit(ada_X_train, ada_y_train)

ada_y_pred = ada_clf.predict(ada_X_test).astype(int)
ada_y_pred2 = ada_clf.predict(ada_X_test2).astype(int)

def plot_search_results(grid):

    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid

    ## Plotting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN F1 SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o')
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()

plot_search_results(ada_grid_search)

# TUNING SCORES
print("Scores after tuning", ada_clf.get_params())
print("Accuracy:",metrics.accuracy_score(ada_y_test, ada_y_pred))
print("The precision score is: %.2f" % metrics.precision_score(ada_y_test, ada_y_pred, pos_label=1))
print("The recall score is: %.2f" % metrics.recall_score(ada_y_test, ada_y_pred, pos_label=1), "\n")
print("The F1 score is: %.2f" % metrics.f1_score(ada_y_test, ada_y_pred, pos_label=1))

cm = metrics.confusion_matrix(ada_y_test, ada_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=ada_clf.classes_)
disp.plot()

plt.figure(figsize=(20, 12))
plt.show()

#%% VISUALISE ADA
tree_pred2 = df_test3_1H.copy() #make another column with target species in all rows for clustering purposes
tree_pred2['y_pred'] = ada_y_pred2
pred_loc2 = tree_pred2[tree_pred2['y_pred'] == 1]
#pred_loc_new2 = pred_loc2[pred_loc2['presence'] == 0]
#for sp in species
#plot
# plot train and test data for a random species
plt.close('all')
plt.figure(0)

# get test locations and plot
# test_inds_pos is the locations where the selected species is predicted to be present
# test_inds_neg is the locations where the selected species is predicted to be absent

# get train locations and plot
train_inds_pos = np.where(train_species(sp)['presence'] == 1)[0]
train_inds_neg = np.where(train_species(sp)['presence'] == 0)[0]
#plt.plot(train_species(sp).iloc[train_inds_pos, 1], train_species(sp).iloc[train_inds_pos, 0], 'rx', label='train postive')
#plt.plot(train_species(sp).iloc[train_inds_neg, 1], train_species(sp).iloc[train_inds_neg, 0], 'kx', label='train negative')


# get test locations and plot
test_inds_pos = np.where(test_species(sp)['presence'] == 1)[0]
test_inds_neg = np.where(test_species(sp)['presence'] == 0)[0]
#plt.plot(test_species(sp).iloc[test_inds_pos, 1], test_species(sp).iloc[test_inds_pos, 0], 'rx', label='test postive')
#plt.plot(test_species(sp).iloc[test_inds_neg, 1], test_species(sp).iloc[test_inds_neg, 0], 'kx', label='test negative')

#Plot prediction
pred_inds_pos = np.where(tree_pred2['y_pred'] == 1)[0]
pred_inds_neg = np.where(tree_pred2['y_pred'] == 0)[0]
plt.plot(tree_pred2.iloc[pred_inds_pos, 1], tree_pred2.iloc[pred_inds_pos, 0], 'y.', label='prediction')
plt.plot(tree_pred2.iloc[pred_inds_neg, 1], tree_pred2.iloc[pred_inds_neg, 0], 'b.', label='negative prediction')


#plt.plot(df_test3.iloc[:, 1], df_test3.iloc[:, 0], 'g.', label='all coordinates')
plt.title('Predicted distribution of ' + sp_name + ' (ID: ' + str(sp) + ')')
plt.grid(True)
plt.xlim([-130, -60])
plt.ylim([20, 50])
plt.ylabel('latitude')
plt.xlabel('longitude')
plt.legend()
plt.show()

#%% GRADIENT BOOSTING
from sklearn.ensemble import GradientBoostingClassifier

df_train = train_species(sp)[cols]
boost_X_train = df_train[features]
boost_y_train = df_train['presence']

df_test = test_species(sp)[cols]
boost_X_test = df_test[features]
boost_X_test2 = df_test3_1H[features]
boost_y_test = df_test['presence']

boost_clf = GradientBoostingClassifier(random_state=42, 
                                       init = tree_clf)
##unhash when classifier is trained
#boost_clf.set_params(**boost_grid_search.best_params_)
boost_clf.fit(boost_X_train, boost_y_train)

boost_y_pred = boost_clf.predict(boost_X_test).astype(int)
boost_y_pred2 = boost_clf.predict(boost_X_test2).astype(int)

print(sp_name, '(ID:' + str(sp) + ")\nMetrics before tuning: "+str(boost_clf.get_params()))
print("Accuracy:",metrics.accuracy_score(boost_y_test, boost_y_pred))
print("The precision score is: %.2f" % metrics.precision_score( boost_y_test, boost_y_pred, pos_label=1))
print("The recall score is: %.2f" % metrics.recall_score( boost_y_test, boost_y_pred, pos_label=1))
print("The F1 score is: %.2f" % metrics.f1_score( boost_y_test, boost_y_pred, pos_label=1))

cm = metrics.confusion_matrix(boost_y_test , boost_y_pred )
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=boost_clf.classes_)
disp.plot()
#%% GB PARAMS GRID

boost_param_grid = {
    'max_depth': np.arange(5, 20, 5),
    'subsample' : np.arange(0.1, 1.2, 0.2),
    'n_estimators' : np.arange(50, 225, 25)
}

boost_grid_search = GridSearchCV(boost_clf, boost_param_grid, verbose=3, scoring= 'f1', cv=3)
boost_grid_search.fit(boost_X_train, boost_y_train)

plot_search_results(boost_grid_search)

#%% ADABOOST PREDICTIONS MAP AS IN REPORT

from sklearn.ensemble import AdaBoostClassifier

sp = 4146
sp_name = get_taxa_by_id(sp)["results"][0].get("name")

ada_best_params = {'estimator__max_depth': np.int64(5), 'learning_rate': 0.1, 'n_estimators': np.int64(50)}

df_train = train_species(sp)[cols]
ada_X_train = df_train[features]
ada_y_train = df_train['presence']

df_test = test_species(sp)[cols]
ada_X_test = df_test[features]
ada_X_test2 = df_test3_1H[features]
ada_y_test = df_test['presence']

ada_clf = AdaBoostClassifier(estimator = tree_clf, 
algorithm = 'SAMME',
random_state=42)
ada_clf.set_params(**ada_best_params)
ada_clf.fit(ada_X_train, ada_y_train)

ada_y_pred = ada_clf.predict(ada_X_test).astype(int)
ada_y_pred2 = ada_clf.predict(ada_X_test2).astype(int)

tree_pred2 = df_test3_1H.copy() #make another column with target species in all rows for clustering purposes
tree_pred2['y_pred'] = ada_y_pred2
pred_loc2 = tree_pred2[tree_pred2['y_pred'] == 1]

#plot
plt.close('all')
plt.figure(0)

#Plot prediction
pred_inds_pos = np.where(tree_pred2['y_pred'] == 1)[0]
pred_inds_neg = np.where(tree_pred2['y_pred'] == 0)[0]
plt.plot(tree_pred2.iloc[pred_inds_pos, 1], tree_pred2.iloc[pred_inds_pos, 0], 'y.', label='prediction')
#plt.plot(tree_pred2.iloc[pred_inds_neg, 1], tree_pred2.iloc[pred_inds_neg, 0], 'b.', label='negative prediction')


plt.title('Predicted distribution of ' + sp_name + ' (ID: ' + str(sp) + ')')
plt.grid(True)
plt.xlim([-130, -60])
plt.ylim([20, 50])
plt.ylabel('latitude')
plt.xlabel('longitude')
plt.legend()
plt.show()

# %% TREE PREDICTION MAP AS IN REPORT
df_train = train_species(sp)[cols]
features = np.delete(df_train.columns[7:], 1)
tree_X_train = df_train[features]
tree_y_train = df_train['presence']

df_test = test_species(sp)[cols]
tree_X_test = df_test[features]
tree_X_test2 = df_test3_1H[features]
tree_y_test = df_test['presence']

tree_clf = DecisionTreeClassifier(random_state=42)
tree_best_params={
    'class_weight': 'balanced',
    'criterion': 'gini',
    'max_depth': np.int64(12),
    'min_samples_leaf': np.int64(4)
}
tree_clf.set_params(**tree_best_params)
tree_clf.fit(tree_X_train, tree_y_train)


tree_y_pred = tree_clf.predict(tree_X_test).astype(int)
tree_y_pred2 = tree_clf.predict(tree_X_test2).astype(int)

tree_pred2 = df_test3_1H.copy() #make another column with target species in all rows for clustering purposes
tree_pred2['y_pred'] = tree_y_pred2
pred_loc2 = tree_pred2[tree_pred2['y_pred'] == 1]
#pred_loc_new2 = pred_loc2[pred_loc2['presence'] == 0]

#plot
plt.close('all')
plt.figure(0)

#Plot prediction
pred_inds_pos = np.where(tree_pred2['y_pred'] == 1)[0]
pred_inds_neg = np.where(tree_pred2['y_pred'] == 0)[0]
plt.plot(tree_pred2.iloc[pred_inds_pos, 1], tree_pred2.iloc[pred_inds_pos, 0], 'y.', label='prediction')


plt.title('Predicted distribution of ' + sp_name + ' (ID: ' + str(sp) + ')')
plt.grid(True)
plt.xlim([-130, -60])
plt.ylim([20, 50])
plt.ylabel('latitude')
plt.xlabel('longitude')
plt.legend()
plt.show()

plt.close('all')
plt.figure(0)

#%% KNOWN DISTRIBUTION AS IN REPORT

test_inds_pos = np.where(test_species(sp)['presence'] == 1)[0]
test_inds_neg = np.where(test_species(sp)['presence'] == 0)[0]
plt.plot(test_species(sp).iloc[test_inds_pos, 1], test_species(sp).iloc[test_inds_pos, 0], 'rx', label='test postive')
plt.plot(test_species(sp).iloc[test_inds_neg, 1], test_species(sp).iloc[test_inds_neg, 0], 'kx', label='test negative')

plt.title('Known distribution of ' + sp_name + ' (ID: ' + str(sp) + ')')
plt.grid(True)
plt.xlim([-130, -60])
plt.ylim([20, 50])
plt.ylabel('latitude')
plt.xlabel('longitude')
plt.legend()
plt.show()
#%% WRITE AdaBoost Results to File
# Write to file
sp_locs = pd.DataFrame(tree_pred2.loc[:,'Lat': 'Cls'])
for col in ['Alt', 'Alt_zone', 'y_pred']:
    sp_locs[col] = tree_pred2.loc[:,col]

sp_locs.to_csv('{0}_AdaBoosted.csv'.format(sp), sep='\t')

