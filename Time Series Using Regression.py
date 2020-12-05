# -*- coding: utf-8 -*-
"""
#same dataset is given on Edureka time series -- vipul
@author: vipul
"""
#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading dataset
ds = pd.read_csv(r"D:\MTech\1st Year\ML\Project\Time-Series-Data\AirPassengers.csv")

#Converting dataset into training and test set 
passr = np.array(ds['#Passengers'])
tp = np.array(range(1,97))

passr_train= passr[:96]
n = len(passr_train)
passr_test = passr[96:]

#Removing outliers using normal dist and SD
mean = np.mean(passr_train,axis =0)
std = np.std(passr_train, axis =0)
list1 = passr_train.copy()
passr_clean = np.zeros(n)
N=2
nsd = N*std 
for x in range(n):
    if list1[x] < (mean-nsd):
        passr_clean[x] = mean - nsd
    elif list1[x] > (mean+nsd):
        passr_clean[x] = mean + nsd
    else:
        passr_clean[x] = list1[x]

plt.plot(list1)
plt.plot(passr_clean) 
plt.show()      

#Smooothen the data using Moving Average
#Smoothening data removes the SEASONAL and IRREGULAR component 
        
movavg = np.zeros(n)
x=0
for i in range(2,n-1):
    movavg[i] = sum(passr_clean[x:x+4])/4
    x+=1
#Centered moving average
cmovavg = np.zeros(n)
x=2
for i in range(2,n-2):
    cmovavg[i] = sum(movavg[x:x+2])/2
    x+=1

plt.plot(cmovavg)
plt.plot(passr_clean,color = 'Red')   
plt.show()
 
#Extracting Seasonality and irregularity from data
#S = Sesonal Component, I = Irregular Component, T=Trend
#TIme Series(Y) = S*I*T
seasonal_irreg = np.zeros(n)
for i in range(n-1):
    if cmovavg[i]!=0: seasonal_irreg[i] = passr_clean[i]/cmovavg[i]
print(seasonal_irreg)

seasonality = np.zeros(n)

for i in range(12):
    j = i
    count=0
    s=0
    while(j<n):
        if seasonal_irreg[j] !=0 : count+=1
        s += seasonal_irreg[j] 
        j=j+12
    x =  s/count
    k = i
    while(k<n):
        seasonality[k]=x
        k+=12    

#Deseasonalize the data (De-Seasonalize = Y/S)
De_season = np.zeros(n)
for i in range(n):
    De_season[i] = passr_clean[i]/seasonality[i]
print(De_season)    

#Generating Trend Using Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #The linear regression model
regressor.fit(tp[:,np.newaxis] , De_season[:,np.newaxis]) #fits the x and y axis on the model
print(regressor.intercept_) #Intercept of the regression line
print(regressor.coef_) 

#Apply Prediction Now and calculate trend
tp2 = np.array(range(1,145))
Trend = regressor.predict(tp2[:,np.newaxis])
Trend

#Final forecast = Seasonality * Trend
seasonalityFuture = np.zeros(144)
for i in range(144):
    seasonalityFuture[i] = seasonality[i%12]
seasonalityFuture
Forecast = np.multiply(seasonalityFuture[: , np.newaxis] , Trend)

#Plotting results - Forecasted value vs actual value
plt.plot(Forecast,color = 'Red')
plt.plot(passr)
plt.show()

#Predicted Forecast for last 4 years
forecast_test=Forecast[-48:]

#Actual Result for last 4 years
passr_test[:,np.newaxis]

#Forecast Bias = (Actual-Forecast)/Forecast-- (for test data)
bias = np.subtract(passr_test[:,np.newaxis], forecast_test)
bias = bias/np.sum(forecast_test)
fbias = np.sum(bias)
print(fbias)

#Forecast Accuracy = 1- Absolute Error/Actual
Abs_error = abs(np.subtract(passr_test[:,np.newaxis] , forecast_test))
ForecastAccuracy = 1 - np.sum(Abs_error)/np.sum(passr_test[:,np.newaxis])
print("Forecast Accuracy in % :",ForecastAccuracy*100)












 