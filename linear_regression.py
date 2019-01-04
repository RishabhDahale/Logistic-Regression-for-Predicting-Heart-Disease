import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Importing data
train_data=pd.read_csv("train_values.csv").drop("patient_id", axis=1)
train_labels=pd.read_csv("train_labels.csv").drop("patient_id", axis=1)
test_data=pd.read_csv("test_values.csv")

headers = train_data.columns.values

continuous_label = ['resting_blood_pressure', 'fasting_blood_sugar_gt_120_mg_per_dl', 
	'serum_cholesterol_mg_per_dl', 'oldpeak_eq_st_depression', 'age', 
	'max_heart_rate_achieved']

discrete_lables=[]
for name in headers:
	if name not in continuous_label:
		discrete_lables.append(name)

# This is to transform the labels that are discrete so that we can use onr-hot-encoding on them
label_transformer = {}
encoded_labels=[]
for label in discrete_lables:
	label_transformer[label]=LabelEncoder()
	train_data['encoded_'+label]=label_transformer[label].fit_transform(train_data[label])
	train_data=train_data.drop(label, axis=1)
	encoded_labels.append('encoded_'+label)

# transforming the continuous columns so that mean becones zero and standard deviation is 1
std=train_data[continuous_label].std()
mean=train_data[continuous_label].mean()
train_data[continuous_label] = (train_data[continuous_label] - mean)/std

# Applying one-hot-encoding to labels and preparing training matrix
enc=OneHotEncoder()
X_train=np.append(np.array(train_data[continuous_label]), 
	np.array(enc.fit_transform(train_data[encoded_labels]).toarray()), axis=1)
y_train=train_labels['heart_disease_present']

# Tranforming discrete training data as per the label encoder previously trained
for label in discrete_lables:
	test_data['encoded_'+label]=label_transformer[label].fit_transform(test_data[label])
	test_data.drop(label, axis=1)

# Transforming continuous test data as per the mean and std previously calculated
test_data[continuous_label] = (test_data[continuous_label] - mean)/std

# Creating matrix for testing
X_test=np.append(np.array(test_data[continuous_label]), 
	np.array(enc.fit_transform(test_data[encoded_labels]).toarray()), axis=1)

lr=LogisticRegression(C=1000)
# Training the Logistic Regression classifier
lr.fit(X_train, y_train)

y_test = lr.predict_proba(X_test)[..., 1]

submission=pd.DataFrame({'patient_id':test_data['patient_id'], 'heart_disease_present':y_test})
submission.to_csv("submission_lr.csv", index=False)
