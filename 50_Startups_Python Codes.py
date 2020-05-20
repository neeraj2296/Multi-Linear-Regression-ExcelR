# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 07:39:58 2020

@author: Neeraj KUmar S J
"""
############################################# Importing the Modules ##############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing
from ml_metrics import rmse
############################################# Importing the dataset ##############################################
#toy = pd.read_csv("E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\ToyotaCorolla.csv")
#Stu = pd.read_csv("E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\Toyotacorolla.csv")
Stu = pd.read_csv("E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\50_Startups.csv,encoding = "ISO-8859-1")
stu = pd.read_csv("E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\50_Startups.csv",encoding = "ISO-8859-1")
stu
stu.describe()
stu.columns = 'RD','Ad','MS','St','Pr'
############################################# Label Encoding the state variable ##################################
Le = preprocessing.LabelEncoder()
stu['St_t'] = Le.fit_transform(stu['St'])
stu = stu.drop('St',axis = 1)
stu.columns = 'RD','Ad','MS','Pr','St'
#finding the correlation between variables
cor_stu = stu.corr()
############################################# Visualizing and normalizing the data to remove outliers  ###########
#sstu = stu.drop('St',axis = 1)
n_stu = preprocessing.normalize(stu)
plt.hist(n_stu[4])
import seaborn as sns
sns.pairplot(stu)

############################################# Building the model1  ################################################

mod1 = smf.ols('Pr~RD+Ad+MS+St',data=stu).fit()
mod1.summary()#Since the p values for Ad, Ms and St are above than 0.05, so lets check for significance

mod1_Ad = smf.ols('Pr~Ad',data=stu).fit()#applying model 1 for only Ad against profit
mod1_Ad.summary()# looks like there is 16.9% of errors in prediction caused by Ad variable

mod1_MS = smf.ols('Pr~MS',data=stu).fit()#applying model 1 for only MS against profit
mod1_MS.summary()#Shows that it is significant

mod1_St = smf.ols('Pr~St',data=stu).fit()#applying model 1 for only St against profit
mod1_St.summary()#looks like there is 48.2% of errors in prediction caused by St variable

import statsmodels.api as sm
sm.graphics.influence_plot(mod1)#checking the data points which are influencing

e_stu = stu.drop(stu.index[[49,48,46,19]],axis = 0)# Looks like 49,48,46,19 data points are influencing, 
                                                   # Hence we remove it.
mod1_new = smf.ols('Pr~RD+Ad+MS',data=e_stu).fit()# Applying model 1 for newly created dataset with variables 
                                                  # Ad, Rd and MS   against profit, removing St as it is not 
                                                  # explaining the profit variables.
mod1_new.summary()#Since the p values for Ad and MS are above than 0.05, so lets check for significance
# Hence we calculate vif values for every varaible against other two variables among RD, Ad and MS
# Whose vif value must be less than 10 for the variable to be significant
rsq_RD = smf.ols('RD~Ad+MS',data=e_stu).fit().rsquared# Ad and MS against RD
vif_RD = 1/(1-rsq_RD)# = 1.19698289102702, Hence significant

rsq_MS = smf.ols('MS~RD+Ad',data=e_stu).fit().rsquared# Ad and RD against MS 
vif_MS = 1/(1-rsq_MS)# = 2.99273159509313, Hence significant

rsq_Ad = smf.ols('Ad~MS+RD',data=e_stu).fit().rsquared# RD and MS against Ad
vif_Ad = 1/(1-rsq_Ad)# = 3.04709935040856, Hence significant

d1 = {'Variables':['RD','MS','Ad'],'VIF':[vif_RD,vif_MS,vif_Ad]}# Combining the vif values wrt its variables
Vif_frame = pd.DataFrame(d1)# To a data frame
Vif_frame

sm.graphics.plot_partregress_grid(mod1_new)#Plotting regression models to check which variables explaining the most

fmod1_new = smf.ols('Pr~RD+MS',data=e_stu).fit()#We shall be removing Ad, even though it has feasible vif values, 
                                                #it does'nt have feasible p values to model 1, Hence the model 1 is
                                                #created without Ad
fmod1_new.summary()# Looks R Squared value of the model and the p values of variable are feasible

pred1 = fmod1_new.predict(e_stu)#Predicting the price using model1
rootmse = rmse(pred1,e_stu.Pr)#calculating the root mean square error
rootmse# = 7076.114277848526
act1 = e_stu.Pr
df = pd.DataFrame(list(zip(pred1, act1)),columns =['Predicted Prices', 'Actual Prices'])
df#created the data set of predicted and actual prices.
#Creating a table for all the Rsquared Values of the diffrent models that was built during correction of influenicing poins in the data set.
values = list([mod1.rsquared,mod1_new.rsquared,fmod1_new.rsquared])
coded_variables = list(['mod1.rsquared','mod1_new.rsquared','fmod1_new.rsquared'])
variables = list(['Model 1','Model 1 New','Final Model 1'])
#R_Squared_value_Of_models = {'Variables':[],'R^2 Value':[]}
Rsquared_model = pd.DataFrame(list(zip(variables,coded_variables,values)),columns = ['Models','Variabels Named in the code','R^Squared Values'])
Rsquared_model#Below is the table that shows how, on removing those outliers, R^Squared Value has improved.
'''
          Models Variabels Named in the code  R^Squared Values
0        Model 1               mod1.rsquared          0.950746
1    Model 1 New           mod1_new.rsquared          0.962343
2  Final Model 1          fmod1_new.rsquared          0.961076
...
