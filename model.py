#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

#Import Dataset
df = pd.read_csv('TSLA.csv')
#print(df.head())

df.dropna(inplace=True)
df.drop_duplicates()

df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

df.drop('Adj Close', axis=1, inplace=True)
#print(df.head())

#print(df.isnull().sum())

y = df[['Close']]
X = df[['Open', 'High', 'Low', 'Volume']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

model = LinearRegression()

model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)

pickle.dump(model, open('teslaModel.pkl', 'wb'))

y_preds = pd.DataFrame({'actual': y_test.squeeze(), 'predicted': y_pred.squeeze()})
print(y_preds)

accuracy = r2_score(y_test,y_pred)*100
print("Accuracy of the model is: ", accuracy)
print('Train Score: ', model.score(X_train, y_train)) 

a = pd.DataFrame({'Open': [25.350000], 'High': [25.974001], 'Low': [25.178667], 'Volume': [107820000]})
a

test = model.predict(a)
print(int(test[0]))