#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

# Data cleaning and preparation

# Load data from CSV into a DataFrame
df = pd.read_csv('data.csv')

# Split dataframes
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

# Reset indexes
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# Set target variable
y_train = (df_train.is_funded).astype('int').values
y_test = (df_test.is_funded).astype('int').values
y_val = (df_val.is_funded).astype('int').values

# Remove target variable from dataframes
del df_train['is_funded']
del df_test['is_funded']
del df_val['is_funded']

# Decision tree
# Initial model
# Convert variables to dicts
train_dicts = df_train.fillna(0).to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

# Fit model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Validation dataset
# Now do the same for the validation and prediction
val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)

# Convert form 2D array to 1D to use ROC AUC
y_pred = dt.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred)
print(f'Initial model ROC AUC: {roc_auc}')

# Tuning
# Main tuning parameters: max_depth, min_samples_leaf
# Final model
# Using tuned features to fit the model
# Full train
df_full_train = df_full_train.reset_index(drop=True)
y_full_train = (df_full_train.is_funded).astype('int').values
del df_full_train['is_funded']

dicts_full_train = df_full_train.fillna(0).to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

y_pred = dt.predict_proba(X_full_train)[:, 1]
auc = roc_auc_score(y_full_train, y_pred)
print(f'Full train ROC AUC: {auc}')

# Export model
model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=40)
output_file = 'model.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
