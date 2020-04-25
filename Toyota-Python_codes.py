# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:31:07 2020

@author: Neeraj Kumar S J
"""
############################################################## Importing the necassary modules ########################################################################################################################################################################################################################################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing
from ml_metrics import rmse
############################################################## Importing the dataset ##################################################################################################################################################################################################################################################################################################################################################
#toy = pd.read_csv("E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\ToyotaCorolla.csv")
#toy = pd.read_csv("E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\Toyotacorolla.csv")
toy = pd.read_csv("E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\Toyotacorolla.csv",encoding = "ISO-8859-1")
toy.columns#Checking the columns which coukd be removed
e_toy = toy.drop(['Id', 'Model', 'Mfg_Month', 'Mfg_Year','Fuel_Type', 'Met_Color', 'Color', 'Automatic', 'Cylinders', 'Mfr_Guarantee', 'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2','Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player','Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio','Mistlamps', 'Sport_Model', 'Backseat_Divider', 'Metallic_Rim','Radio_cassette', 'Tow_Bar'],axis = 1)
e_toy.describe
cor_toy = e_toy.corr()#checking the correlation between variables
cor_toy.columns
e_toy.columns = 'Price','Age','KM','HP','cc','Dr','gr','Qt','Wt'#Editting the column names
import seaborn as sns
sns.pairplot(e_toy)#Checking for highly colinear variables
############################################################## BUilding Model 1 #######################################################################################################################################################################################################################################################################################################################################################
mod1 = smf.ols('Price ~ Age+KM+HP+cc+Dr+gr+Qt+Wt',data=e_toy).fit()
mod1.summary()
#Since the p values for cc and Dr are above than 0.05, so lets check for significance
mod_1c = smf.ols('Price ~ cc',data=e_toy).fit()#applying model 1 for only cc against price
mod_1c.summary()#Shows that it is significant

mod_1d =  smf.ols('Price~Dr',data=e_toy).fit()#applying model 1 for only Dr against price
mod_1d.summary()#Shows that it is significant

mod1_dc = smf.ols('Price~cc+Dr',data=e_toy).fit()#applying model 1 for Dr and cc against price
mod1_dc.summary()#Shows that both are significant

import statsmodels.api as sm
sm.graphics.influence_plot(mod1)#checking the data points which are influencing

e_new = e_toy.drop(e_toy.index[[80,960,221]],axis = 0)#Looks like 80,860,221 data points are influencing, Hence we remove it.

mod1_new = smf.ols('Price ~ Age+KM+HP+cc+Dr+gr+Qt+Wt',data=e_new).fit()#Applying model1 for the newly created data set
mod1_new.summary()#Looks good here as all the variable's p values are below 0.05
act1 = e_new.Price
#sm.graphics.plot_partregress_grid(mod1_new)
#Prdicting Prices using mod1
pred1 = mod1_new.predict(e_new)#Predicting the price using model1
rootmse = rmse(pred1,e_new.Price)#calculating the root mean square error
rootmse# = 1227.473986005888
df = pd.DataFrame(list(zip(pred1, act1)),columns =['Predicted Prices', 'Actual Prices'])#creating the data set of predicted and actual prices.
df
'''
############################################################## BUilding Model 2 #######################################################################################################################################################################################################################################################################################################################################################
mod2 = smf.ols('Price ~ np.log(Age)+KM+HP+cc+Dr+gr+Qt+Wt',data=e_new).fit()
mod2.summary()
#Since the p values for cc and Dr are above than 0.05, so lets check for significance
mod_2d =  smf.ols('Price~Dr',data=e_new).fit()#applying model 2 for only Dr against price
mod_2d.summary()#Shows that it is significant
#e_new2 = e_new2.drop(['Dr'], axis = 1)

#import statsmodels.api as sm
sm.graphics.influence_plot(mod2)#checking the data points which are influencing

e_new2 = e_new.drop(e_new.index[[184,185,991,956,109,110,111,49]],axis = 0)#Looks like 184,185,991,956,109,110,111,49 data points are influencing, Hence we remove it.

mod2_new = smf.ols('Price ~ np.log(Age)+KM+HP+cc+Dr+gr+Qt+Wt',data=e_new2).fit()#Applying model2 for the newly created data set
mod2_new.summary()#Looks Dr variable values are not yet below 0.05

sm.graphics.influence_plot(mod2_new)#checking the data points which are influencing

#e_new3 = e_new2.drop(e_new2.index[[184,185,991,956,109]],axis = 0)#Looks like 184,185,991,956,109 data points are influencing, Hence we remove it.
sm.graphics.plot_partregress_grid(mod2_new)

fmod2_new = smf.ols('Price ~ np.log(Age)+KM+HP+cc+gr+Qt+Wt',data=e_new2).fit()#Applying model2 for the newly created data set and we shal remove Dr Variable
fmod2_new.summary()#Looks good here as all the variable's p values are below 0.05

#Prdicting Prices using mod2
pred2 = mod2_new.predict(e_new2)
rootmse2 = rmse(pred2,e_new2.Price)
rootmse2 # = 1256.1065020469682  <<Seems like model2 is better than model 1, based on Root Mean Square 

act2 = e_new2.Price
df = pd.DataFrame(list(zip(pred2, act2)),columns =['Predicted Prices', 'Actual Prices'])#creating the data set of predicted and actual prices. 
df
sns.pairplot(e_new3);sns.pairplot(e_new.Prices, pred2, color='black')

############################################################## BUilding Model 3 #######################################################################################################################################################################################################################################################################################################################################################
mod3 = smf.ols('Price ~ np.log(Age)+np.log(KM)+HP+cc+gr+Qt+Wt',data=e_new3).fit()
mod3.summary()
#Since the p values of all variables are less than 0.05. We can go for prediction
#Prdicting Prices using mod3
pred3 = mod3.predict(e_new3)
rootmse3 = rmse(pred3,e_new3.Price)
rootmse3 # = 1356.442261894533  <<Seems like model3 is not better than model 1 or 2, based on Root Mean Square , but it is better in terms or Rsquared value
act3 = e_new3.Price
df = pd.DataFrame(list(zip(pred3, act3)),columns =['Predicted Prices', 'Actual Prices'])#creating the data set of predicted and actual prices. 
df
#sns.pairplot(e_new3);sns.pairplot(e_new.Prices, pred2, color='black')
############################################################## BUilding Model 4 #######################################################################################################################################################################################################################################################################################################################################################

mod4 = smf.ols('Price ~ np.log(Age)+np.log(KM)+np.log(cc)+HP+gr+Qt+Wt',data=e_new3).fit()
mod4.summary()
#Since the p values of all variables are less than 0.05. We can go for prediction
#Prdicting Prices using mod3
pred4 = mod4.predict(e_new3)
rootmse4 = rmse(pred4,e_new3.Price)
rootmse4 # = 1356.442261894533  <<Seems like model3 is not better than model 1 or 2, based on Root Mean Square , but it is better in terms or Rsquared value
act4 = e_new3.Price
df = pd.DataFrame(list(zip(pred4, act4)),columns =['Predicted Prices', 'Actual Prices'])#creating the data set of predicted and actual prices. 
df
############################################################## BUilding Model 5 #######################################################################################################################################################################################################################################################################################################################################################

mod5 = smf.ols('Price ~ np.log(Age)+np.log(KM)+np.log(cc)+HP+np.log(gr)+Qt+Wt',data=e_new3).fit()
mod5.summary()
#Since the p values of all variables are less than 0.05. We can go for prediction
#Prdicting Prices using mod3
pred5 = mod5.predict(e_new3)
rootmse5 = rmse(pred5,e_new3.Price)
rootmse5 # = 1356.442261894533  <<Seems like model3 is not better than model 1 or 2, based on Root Mean Square , but it is better in terms or Rsquared value
act5 = e_new3.Price
df = pd.DataFrame(list(zip(pred5, act5)),columns =['Predicted Prices', 'Actual Prices'])#creating the data set of predicted and actual prices. 
df
'''