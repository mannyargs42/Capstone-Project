#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.cross_validation import H2OKFold
from h2o.model.regression import h2o_r2_score
from h2o.model.metrics_base import H2OBinomialModelMetrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from h2o.model.regression import h2o_mean_squared_error
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator


# In[2]:


# open the cleaned csv file from data prep and variable extraction
df = pd.read_csv('pred_ready.csv')

# open the outcomes from the original dataset
outcomes = pd.read_csv('outcomes_a.csv')

# drop the extra column from each
df.drop('Unnamed: 0', axis=1, inplace=True)
outcomes.drop('Unnamed: 0', axis=1, inplace=True)

# set outcomes as series y
y = outcomes['In-hospital_death']

# adjust outcomes for to 'survived' and in-hospital-death ('IHD')
survival = []
for x in y:
    if x == 0:
        survival.append('SURVIVED')
    if x == 1:
        survival.append('IHD')

y = pd.DataFrame(survival, columns=['In-hospital_death'])

y.head(10)


# ## Scikit-learn
# #### Imputing values and baseline random forest

# In[3]:


# impute means for random forest classifier
# standardize data using standard scaler
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
scaler = StandardScaler()
df1 = imp_mean.fit_transform(df.drop('RecordID', axis=1))
df1_ = scaler.fit_transform(df1)
# use sklearn's train test split to have train and validation set
X_train, X_test, y_train, y_test = train_test_split(df1_, y, test_size=0.25,
                                                   random_state=42)

# create random forest classifier
ranfor = RandomForestClassifier(random_state=42)

# fit classifier and predict on dataset
ranfor.fit(X_train, y_train)
y_pred = ranfor.predict(X_test)

# get accuracy score for predictions
acc = metrics.accuracy_score(y_test, y_pred)
acc


# ## H2O
# #### Data munging

# In[4]:


# h2o.init to initiate h2o

h2o.init()

# we'll need column headers after converting pandas df to h2o df
# upload pred_ready file and outcomes
df_h2o = h2o.upload_file('pred_ready.csv')
print(df_h2o.shape)
oc_h2o = h2o.upload_file('outcomes_a.csv')

# see how imbalanced label class is
print(oc_h2o['In-hospital_death'].table())

# percentage-wise imbalance:
n = df_h2o.shape[0]
print(oc_h2o['In-hospital_death'].table()['Count'] / n)

# set up labels column
y_h2o = oc_h2o['In-hospital_death']
print(y_h2o)

# set up predictors dataframe
# drop the RecordID column and initial index column
X = df_h2o.drop([0,1])
print(X)

# get column headers from X to use as labels for dataframe
x_n = X.columns

# create dataframe from df1 with imputed means, then correct headers
df2 = h2o.H2OFrame(df1_)
df2.columns=x_n
print(df2)


# In[5]:


# convert y as categorical outcomes to h2o dataframe
y = h2o.H2OFrame(y)

# add to h2o dataframe to keep instances and labels aligned
df_fullh2o = df2.concat(y)

# remove columns that contain only one value and don't contribute to
    # predictions
X_h2o = df_fullh2o.drop(['MechVent_last', 'MechVent_median', 
                     'MechVent_q3', 'MechVent_mean', 'MechVent_q1', 
                     'MechVent_min', 'MechVent_max', 
                     'MechVent_first'])
print(X_h2o)

# split data into test set and validation set
full_train, full_test = X_h2o.split_frame(ratios = [0.7493], seed=41)
print(full_train.shape)
print(full_test.shape)

# drop labels column from train and validation sets
X_trainh2o = full_train.drop([330])
X_testh2o = full_test.drop([330])

# convert labels to categorical then to string types for both sets
y_trainh2o = full_train['In-hospital_death'].asfactor()
y_traindl = y_trainh2o.asnumeric()
y_testh2o = full_test['In-hospital_death'].asfactor()
y_testdl = y_testh2o.asnumeric()


# In[35]:


# before training, get an ideal of class imbalance

oc_h2o['In-hospital_death'].table()

oc_pd = pd.DataFrame({'Outcome': ['Survival', 'Mortality'],
                     'Count': [3446, 554]})

oc_pd

plt.bar(oc_pd['Outcome'], oc_pd['Count'], color="teal")
plt.title('Class Imbalance', fontsize=16)
plt.xlabel('Mortality Rates')
plt.ylabel('Count')
plt.show()


# ## H2O
# #### ML algorithm grid searches

# In[6]:


gb_params = {
        'ntrees': [10,50,100], 
        'nfolds': [0, 3, 5],    
}

ranfor_params = {
        'ntrees': [5, 10, 50], 
        'nfolds': [0, 3, 5]  
}

dl_params = {
        'nfolds': [0, 3, 5],
        'train_samples_per_iteration': [100, 1100, 1500],
        'epochs': [5, 10]    
}


# In[7]:


gb = H2OGradientBoostingEstimator()
ranfor = H2ORandomForestEstimator()
dl = H2ODeepLearningEstimator()


# In[8]:


custom = H2OKFold(X_trainh2o, n_folds=5, seed=42)


# In[9]:


# check the output types, dl needs to have real numbers

print(y_trainh2o.types)
print(y_traindl.types)


# In[10]:


# set up randomized search cv for each classifier
gbsearch = RandomizedSearchCV(gb, gb_params, n_iter=5,
                              scoring=make_scorer(h2o_r2_score),
                              cv=custom, random_state=42)
ransearch = RandomizedSearchCV(ranfor, ranfor_params, n_iter=5,
                              scoring=make_scorer(h2o_r2_score),
                              cv=custom, random_state=42)
dlsearch = RandomizedSearchCV(dl, dl_params, n_iter=5,
                              scoring=make_scorer(h2o_r2_score),
                              cv=custom, random_state=42)


# In[12]:


# run randomized search cv for gb, ranfor, and dl estimators
gbsearch.fit(X_trainh2o, y_trainh2o)
ransearch.fit(X_trainh2o, y_trainh2o)
dlsearch.fit(X_trainh2o, y_traindl)

print('GB best: {}'.format(gbsearch.best_params_))
print('RF best: {}'.format(ransearch.best_params_))
print('DL best: {}'.format(dlsearch.best_params_))


# # Stacked Ensemble formation

# In[11]:


x_t = X_h2o.columns


# In[12]:


y_t = 'In-hospital_death'


# In[13]:


# use best results from randomized search cv to create classifiers
# then create a stacked ensemble of the three

h2o_gb = H2OGradientBoostingEstimator(ntrees=50, nfolds=5,
                                    fold_assignment='modulo', seed=42,
                                     keep_cross_validation_predictions=True)

h2o_ranfor = H2ORandomForestEstimator(ntrees=10, nfolds=5, 
                                    fold_assignment='modulo', seed=42,
                                     keep_cross_validation_predictions=True)

h2o_deep = H2ODeepLearningEstimator(activation='tanh',
                                    nfolds=5, train_samples_per_iteration=100, 
                                    fold_assignment='modulo', shuffle_training_data=True, 
                                    epochs=5, missing_values_handling='skip', 
                                    distribution='auto', seed=42, 
                                    keep_cross_validation_predictions=True)


# the 3 base models have to be trained prior to ensemble formation
h2o_gb.train(x=x_t, y=y_t, training_frame=full_train, validation_frame=full_test)
h2o_ranfor.train(x=x_t, y=y_t, training_frame=full_train, validation_frame=full_test)
h2o_deep.train(x=x_t, y=y_t, training_frame=full_train, validation_frame=full_test)

model_list = [h2o_gb, h2o_ranfor, h2o_deep]
stack = H2OStackedEnsembleEstimator(model_id='stackd', 
                                        base_models=model_list,
                                      metalearner_nfolds=5, seed=42)

stack.train(x=x_t, y=y_t, training_frame=full_train, validation_frame=full_test)


# In[34]:


# check model performances on validation data and cross-validation

print(h2o_gb)
print(h2o_ranfor)
print(h2o_deep)
print(stack)


# ## Model Evaluation
# #### Evaluate each model based on predictions on Set B

# In[51]:


# repeat all data preparation steps for set b

B = pd.read_csv('pred_ready_b.csv')
outcomes_b = pd.read_csv('outcomes_b.csv')
print(B)
print(outcomes_b)
B.drop('Unnamed: 0', axis=1, inplace=True)
outcomes.drop('Unnamed: 0', axis=1, inplace=True)
y_ = outcomes['In-hospital_death']
survival = []
for x in y_:
    if x == 0:
        survival.append('SURVIVED')
    if x == 1:
        survival.append('IHD')
y_t = pd.DataFrame(survival, columns=['In-hospital_death'])
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
scaler = StandardScaler()
df_t = imp_mean.fit_transform(B.drop('RecordID', axis=1))
df_t_ = scaler.fit_transform(df_t)


df_b = h2o.H2OFrame(df_t_)
df_b.columns=x_n

# convert y as categorical outcomes to h2o dataframe
y_b = h2o.H2OFrame(y_t)

# add to h2o dataframe to keep instances and labels aligned
df_fullh2o = df_b.concat(y_b)

# remove columns that contain only one value and don't contribute to
    # predictions
X_h2o = df_fullh2o.drop(['MechVent_last', 'MechVent_median', 
                     'MechVent_q3', 'MechVent_mean', 'MechVent_q1', 
                     'MechVent_min', 'MechVent_max', 
                     'MechVent_first'])
print(X_h2o)


# In[42]:


# run predictions on each of the models

h2o_gb.predict(X_h2o)
h2o_ranfor.predict(X_h2o)
h2o_deep.predict(X_h2o)
stack.predict(X_h2o)


# In[46]:


print(h2o_gb.model_performance(X_h2o))
print(h2o_ranfor.model_performance(X_h2o))
print(h2o_deep.model_performance(X_h2o))
print(stack.model_performance(X_h2o))


# In[318]:


h2o_gb.model_performance(full_test)


# In[320]:


h2o_gb.auc()


# In[168]:


oc_h2o['In-hospital_death'].table()

oc_pd = pd.DataFrame({'Outcome': ['Survival', 'Mortality'],
                     'Count': [3446, 554]})

oc_pd

plt.bar(oc_pd['Outcome'], oc_pd['Count'], color="teal")
plt.title('Class Imbalance', fontsize=16)
plt.xlabel('Mortality Rates')
plt.ylabel('Count')
plt.show()


# In[323]:


deep_pred


# In[325]:


df_deep = deep_pred.as_data_frame()


# In[333]:


h2o_gb.sensitivity(valid=True)


# In[331]:


h2o_gb.confusion_matrix()


# In[335]:


h2o_ranfor.model_performance(full_test)


# In[334]:


h2o_gb.model_performance(full_test)


# In[336]:


h2o_deep.model_performance(full_test)


# In[337]:


ensemble.model_performance(full_test)


# In[327]:


d = df_deep['predict'].tolist()


# In[ ]:




