# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:17:06 2020

@author: Neeraj Kumar S J
"""

############################################# Importing the Modules ##############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing
from ml_metrics import rmse

############################################# Importing the dataset ##############################################

computer = pd.read_csv("E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\Computer_Data.csv")
comp = computer
comp = comp.drop(['Unnamed: 0'],axis = 1)
comp.columns = ['pr','sp','hd','ram','sc','cd','mu','pre','ad','tr']
comp.describe()
############################################# Label Encoding the state variable ##################################
Le = preprocessing.LabelEncoder()
comp['cd_t'] = Le.fit_transform(comp['cd'])
comp['mu_t'] = Le.fit_transform(comp['mu'])
comp['pre_t'] = Le.fit_transform(comp['pre'])
comp = comp.drop('cd',axis = 1)
comp = comp.drop('mu',axis = 1)
comp = comp.drop('pre',axis = 1)
comp.columns = ['pr','sp','hd','ram','sc','ad','tr','cd','mu','pre']

#finding the correlation between variables
cor_comp = comp.corr()

############################################# Visualizing and normalizing the data to remove outliers  ###########
#scomp = comp.drop('St',axis = 1)
plt.hist
n_comp = preprocessing.normalize(comp)
plt.hist(n_comp)
import seaborn as sns
sns.pairplot(comp)
############################################# Building the model 1  ###############################################

mod1 = smf.ols('pr~sp+hd+ram+sc+ad+tr+cd+mu+pre',data=comp).fit()
mod1.summary()
import statsmodels.api as sm
sm.graphics.influence_plot(mod1)

e_comp = comp.drop(comp.index[[5960,1101,900]],axis = 0)

nmod1 = smf.ols('pr~sp+hd+ram+sc+ad+tr+cd+mu+pre',data=e_comp).fit()
nmod1.summary()

pred1 = mod1.predict(e_comp)#Predicting the price using model1
rootmse1 = rmse(pred1,e_comp.pr)#calculating the root mean square error
rootmse1# = 274.9527714849643
act1 = comp.pr
df = pd.DataFrame(list(zip(pred1, act1)),columns =['Predicted Prices', 'Actual Prices'])
df#created the data set of predicted and actual prices.

############################################# Building the model 2  ###############################################

mod2 = smf.ols('pr~np.log(sp)+np.log(hd)+ram+sc+ad+tr+cd+mu+pre',data=e_comp).fit()
mod2.summary()

#import statsmodels.api as sm
sm.graphics.influence_plot(mod2)
e0_comp = e_comp.drop(e_comp.index[[1400,1700,79,85,3,169,230,1688,2281]],axis = 0)

mod2_new = smf.ols('pr~np.log(sp)+np.log(hd)+ram+sc+ad+tr+cd+mu+pre',data=e0_comp).fit()
mod2_new.summary()

sm.graphics.influence_plot(mod2_new)

e1_comp = e0_comp.drop(e0_comp.index[[6185,54,1991,4755,4125,4354,141,2474]],axis = 0)

fmod2 = smf.ols('pr~np.log(sp)+np.log(hd)+ram+sc+ad+tr+cd+mu+pre',data=e1_comp).fit()
fmod2.summary()

pred2 = fmod2.predict(e1_comp)#Predicting the price using model1
rootmse2 = rmse(pred2,e1_comp.pr)#calculating the root mean square error
rootmse2# = 256.839537400504
act2 = comp.pr
df = pd.DataFrame(list(zip(pred2, act2)),columns =['Predicted Prices', 'Actual Prices'])
df#created the data set of predicted and actual prices.

#mod2 = smf.ols('pr~np.log(sp)+np.log(hd)+ram+sc+ad+tr+cd+mu+pre',data=comp).fit()
#mod2.summary()


#sm.graphics.influence_plot(mod2)
#e_comp = comp.drop(comp.index[[1400,1700]],axis = 0)

############################################# Building the model 3  ################################################

e1_comp['ad_sq'] = np.square(e1_comp.ad)
e1_comp['pre_sq'] = np.square(e_comp.pre)
mod3 = smf.ols('np.log(pr)~np.log(sp)+np.log(hd)+ram+sc+ad_sq+tr+cd+mu+pre',data=e1_comp).fit()
mod3.summary()

sm.graphics.influence_plot(mod3)

e2_comp = e1_comp.drop(e1_comp.index[[5429,5373,604,5075,4853,4685,4648,4268,5349,4066,4363,4227,4259,3990,4209,4282,4073,4091,3535,3767,4003,3964,981,5434,3828,4005,5345,271,5423,645,5212,3183,5452,3821,70,4212,3666,2256,2232,1856]],axis = 0)
                                   

mod3_new = smf.ols('np.log(pr)~np.log(sp)+np.log(hd)+ram+sc+ad_sq+tr+cd+mu+pre',data=e2_comp).fit()
mod3_new.summary()

sm.graphics.influence_plot(mod3_new)

e3_comp = e2_comp.drop(e2_comp.index[[1440,1688,2281,4409,4091,3964,981,5345,5434,271,4066,4363,4227,4259,4282,4073,5429,5373,4685,4755,3821,4003,3535,3767,4209,3990,3479,141,4853,5212,3183,5349,5075,4409,3935,2976,1101,1700,1805,309,207,27,174,24,418,313,1117,1047,795,2000,1432,1792]],axis = 0)

final_mod3 = smf.ols('np.log(pr)~np.log(sp)+np.log(hd)+ram+sc+ad_sq+tr+cd+mu+pre',data=e3_comp).fit()
final_mod3.summary()

pred1_log = final_mod3.predict(e3_comp)#Predicting the price using model1
pred1 = np.exp(pred1)
rootmse = rmse(pred1,e3_comp.pr)#calculating the root mean square error
rootmse# = 243.59066549716243

#Since the root mean Square Error is much lesser than that of the model2's root mean square error .
#Model 3 has a better RSquared Value, i.e. R^Squared Error does'nt improve  above 0.815, Hence We Shall Stop
act1 = e3_comp.pr
df = pd.DataFrame(list(zip(pred1, act1)),columns =['Predicted Prices', 'Actual Prices'])
df#created the data set of predicted and actual prices.


values = list([mod1.rsquared,nmod1.rsquared,mod2_new.rsquared,mod2_new.rsquared,fmod2.rsquared,mod3.rsquared,mod3_new.rsquared,final_mod3.rsquared])
variables = list(['Model 1','Model 1 New','Final Model 2','Model 2 New','Final Model 2','Model 3','Model 3 New','Final Model 3'])
coded_variables = list(['mod1.rsquared','nmod1.rsquared','mod2_new.rsquared','mod2_new.rsquared','fmod2.rsquared','mod3.rsquared','mod3_new.rsquared','final_mod3.rsquared'])
Rsquared_model = pd.DataFrame(list(zip(variables,coded_variables,values)),columns = ['Models','Variabels Named in the code','R^Squared Values'])
Rsquared_model#Below is the table that shows how, on removing those outliers, R^Squared Value has improved.
'''
          Models Variabels Named in the code  R^Squared Values
0        Model 1               mod1.rsquared          0.775568
1    Model 1 New              nmod1.rsquared          0.775270
2  Final Model 2           mod2_new.rsquared          0.803485
3    Model 2 New           mod2_new.rsquared          0.803485
4  Final Model 2              fmod2.rsquared          0.803731
5        Model 3               mod3.rsquared          0.815282
6    Model 3 New           mod3_new.rsquared          0.815236
7  Final Model 3         final_mod3.rsquared          0.815467
'''