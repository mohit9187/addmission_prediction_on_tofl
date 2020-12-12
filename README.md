# addmission_prediction_on_tofl
import numpy as np
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("Admission_Prediction.csv")
df.isnull().sum()
df['GRE Score'].fillna(df['GRE Score'].mode()[0],inplace = True)
df['TOEFL Score'].fillna(df['TOEFL Score'].mode()[0],inplace = True)
df['University Rating'].fillna(df['University Rating'].mean(),inplace = True)
x = df.drop(['Serial No.','Chance of Admit'],axis = 1)
y = df['Chance of Admit']
for col in x.columns :
    if (col != ['Chance of Admit']) :
        plt.scatter(x[col],y)
        plt.xlabel("Addmission_chance")
        plt.show()
from sklearn.preprocessing import StandardScaler
Scaler_feature =StandardScaler()
Scaler_label =StandardScaler()
Scaled_data =Scaler_feature.fit_transform(x)       
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.33,random_state =100)
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(train_x,train_y)
from sklearn.metrics import r2_score
score = r2_score(reg.predict(test_x),test_y)
score
