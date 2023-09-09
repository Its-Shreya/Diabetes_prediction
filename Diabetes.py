#!/usr/bin/env python
# coding: utf-8

# In[28]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
data=pd.read_csv(r"C:\Users\shrey\Downloads\Healthcare-Diabetes.csv")
df=data.copy()
print(df)
data.describe().T
data.info()
data.isnull().sum()
data.duplicated().sum()
num_col = ['Pregnancies','Glucose','BloodPressure','SkinThickness'
           ,'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
# Define the function
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")
    print("***********")

# Call the function for each numerical column
for col in num_col:
    target_summary_with_num(df, "Outcome", col)


num_col = ['Pregnancies','Glucose','BloodPressure','SkinThickness'
           ,'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
no_outlier = df
for i in num_col:
    lower_limit = df[i].quantile(0.5)
    upper_limit = df[i].quantile(0.95)
    no_outlier[i] = no_outlier[i].clip(lower_limit, upper_limit)

df.plot(kind = "box" , subplots = True , figsize = (15,15) , layout = (5,5))
df.drop(['Id'] , axis=1 , inplace=True)

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

Sc=StandardScaler()
X_train_Scaled=Sc.fit_transform(X_train)
X_test_Scaled=Sc.fit_transform(X_test)

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.ensemble import BaggingClassifier

bc = BaggingClassifier(n_estimators=150, random_state=2)
bc.fit(X_train,y_train)

y_pred=bc.predict(X_test)

from tensorflow.keras.models import save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
    # Create a Keras Sequential model (a simple example)
keras_model = Sequential()
keras_model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
keras_model.add(Dense(1, activation='sigmoid'))

# Compile the Keras model (customize the loss and optimizer as needed)
keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Keras model with your data (customize the number of epochs)
keras_model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the Keras model as an .h5 file
save_model(keras_model, 'diabetes_model.h5')
import joblib
from sklearn.ensemble import RandomForestClassifier  # Replace with your model import
     

joblib.dump(bc, 'C:\MediHacks\diabetes_model.pkl', compress=('zlib', 3))






# In[ ]:




