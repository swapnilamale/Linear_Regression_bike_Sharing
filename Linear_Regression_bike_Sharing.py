# -*- coding: utf-8 -*-

# linear regression
# dataset: bike sharing

# import libraries
import pandas as pd
import numpy as np

# scikit library for linear regression
import statsmodels.api as sm
from statsmodels.formula.api import ols # for anova test

from sklearn.model_selection import train_test_split, KFold

# visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import pylab

from sklearn.metrics import mean_squared_error

## read the data
path="F:/aegis/4 ml/dataset/supervised/regression/bikesharing/bikesharingdata.csv"
bike=pd.read_csv(path)
print(bike.head())
bike.tail()

# check data types
bike.dtypes 

### list of functions
def splitcols(data):
    nc=data.select_dtypes(exclude='object').columns.values
    fc=data.select_dtypes(include='object').columns.values
    
    return(nc,fc)

# function to plot the histogram, correlation matrix, boxplot based on the chart-type
def plotdata(data,nc,ctype):
    if ctype not in ['h','c','b']:
        msg='Invalid Chart Type specified'
        return(msg)
    
    if ctype=='c':
        cor = data[nc].corr()
        cor = np.tril(cor)
        sns.heatmap(cor,vmin=-1,vmax=1,xticklabels=nc,
                    yticklabels=nc,square=False,annot=True,linewidths=1)
    else:
        COLS = 2
        ROWS = np.ceil(len(nc)/COLS)
        POS = 1
        
        fig = plt.figure() # outer plot
        for c in nc:
            fig.add_subplot(ROWS,COLS,POS)
            if ctype=='b':
                sns.boxplot(data[c],color='yellow')
            else:
                sns.distplot(data[c],bins=20,color='green')
            
            POS+=1
    return(1)

# split the dataset into train and test in the ratio (default=70/30)
def splitdata(data,y,ratio=0.3):
    
    trainx,testx,trainy,testy = train_test_split(data.drop(y,1),
                                                 data[y],
                                                 test_size = ratio )
    
    return(trainx,testx,trainy,testy)
### end of functions ###

# since temp and atemp have a high +ve correlation (0.99), drop 'atemp' from the dataset
bike.drop(columns='atemp',inplace=True)

# verify the change
bike.columns

nc,fc = splitcols(bike)
print(nc)
print(fc)

# check for nulls
bike.isnull().sum()
bike.info()

# check for 0 in numeric columns
bike[nc][bike[nc]==0].count()

# plot the graphs
plotdata(bike,nc,'b') # boxplot
plotdata(bike,nc,'h') # histogram
plotdata(bike,nc,'c') # correlation

# decribe the numeric data
descr = bike[nc].describe()
for c in nc:
    print("numeric column = ", c)
    print(descr[c])
    print(" --- ")
    
# refresh nc,fc
nc,fc = splitcols(bike)
nc
fc
# factor variables
for c in fc:
    print('Factor variable = ', c)
    print(bike[c].unique())
    print(' --- ')
    
# convert the factor values into numbers (acc to the documentation)

### season ###
bike.season = bike.season.replace({'springer':1,'summer':2,'fall':3,'winter':4})
bike.season.unique()

### mnth ###
bike.mnth = bike.mnth.replace({'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12})
bike.mnth.unique()    

### holiday ###
bike.holiday = bike.holiday.replace({'H':1,'NH':0})
bike.holiday.unique()

### workingday ###
bike.workingday = bike.workingday.replace({'N':1, 'Y':0})
bike.workingday.unique()

# for a linear regression, data types of features should be numeric
bike.dtypes

# first model (m1): split the data into train and test
trainx1,testx1,trainy1,testy1 = splitdata(bike,'target')
print(trainx1.shape, trainy1.shape, testx1.shape, testy1.shape)

# --------------------------------------- #
# build the linear regression model (M1)
# --------------------------------------- #
# since the default OLS method doesnt generate 'a' in summary, create a dummy constant value for trainx and testx

trainx1.head(3)
trainx1 = sm.add_constant(trainx1)
trainx1.head(3) 

# Ordinary Least Square
m1 = sm.OLS(trainy1,trainx1).fit()

# summarise the model
m1.summary()

# validating the assumptions
# 1) mean of residuals = 0
np.mean(m1.resid)

# 2) residuals have a constant variance (homoscedasticity)
sns.residplot(m1.resid,m1.predict(),color='blue',lowess=True)

# Hypothesis testing for Heteroscedasticity
# -----------------------------------------
# H0: homoscedasticity
# H1: heteroscedasticity

# i) Breusch-Pagan test against heteroscedasticity
# -----------------------------------------------
import statsmodels.stats.api as stats
# 2nd output value is the pvalue of the Hyp test
pvalue_bp = stats.het_breuschpagan(m1.resid,m1.model.exog)[1]

if pvalue_bp < 0.05:
    print('Reject H0 : Model is Heteroscedastic')
else:
    print('FTR H0: Model is Homoscedastic')
    
# ii) White's test
# -----------------
from statsmodels.stats.diagnostic import het_white
pvalue_wt = het_white(m1.resid,m1.model.exog)[1]

if pvalue_wt < 0.05:
    print('Reject H0 : Model is Heteroscedastic')
else:
    print('FTR H0: Model is Homoscedastic')

### since the model is heteroscedastic, to fix it, we can transform X or Y into another transformation scale ###

# 3) Residuals have a Normal distribution
import scipy.stats as spst
spst.probplot(m1.resid,dist='norm',plot=pylab)

# alternatively, plot a histogram of the residuals to check the distribution
plt.hist(m1.resid)
plt.title('Distribution of Errors from Model 1')

# 4) rows > columns
trainx1.shape

# 5) outliers
# check outliers from the boxplot and interpret it accordingly


## Cross validation ##
cv_mse = []
X = trainx1.values
Y = trainy1.values

folds = 5
kf = KFold(folds)

for train_index,test_index in kf.split(X):
    cv_trainx,cv_trainy = X[train_index], Y[train_index]
    cv_testx, cv_testy = X[test_index], Y[test_index]
    
    cv_model = sm.OLS(cv_trainy,cv_trainx).fit()
    cv_pred = cv_model.predict(cv_testx)
    cv_mse.append(mean_squared_error(cv_testy,cv_pred))
    
# print the cross validation MSE
print(cv_mse)
# mean CV error
cv_mean = np.mean(cv_mse)    
cv_mean

# Cross-Validation Mean Squared Error
print("Cross Validation \n\tmse={}, \n\trmse={}".format(round(cv_mean,2),round(np.sqrt(cv_mean),2)))
    
# /Cross validation

# predictions
testx1 = sm.add_constant(testx1)
p1 = m1.predict(testx1)
p1[0:9]
testy1.head(10)

# compare the Actual Y with the Predicted Y value
df1 = pd.DataFrame({'actual':testy1, 'predicted':round(p1,0)})
df1

# Mean Squared Error
mse1 = mean_squared_error(testy1,p1)
print("Model 1 \n\tmse={}, \n\trmse={}".format(round(mse1,2),round(np.sqrt(mse1),2)))

# plot the actual and predicted values
sns.regplot(testy1,p1,marker='.',color='yellow',line_kws={'color':'red'},ci=None)

# chart that shows actual and predicted values - dist plot
ax1 = sns.distplot(testy1,hist=False,color='red',label='Actual')
sns.distplot(p1,hist=False,color='blue',label='Predicted',ax=ax1)


bike.head(3)
bike.season.unique()
bike.mnth.unique()


# build a model by transforming the Y-variable
# (take log of the y-variable); build the model and predict)

# for the next model, create a new trainx and testx datasets
trainx2 = trainx1.drop('mnth',1)
testx2 = testx1.drop('mnth',1)

'''
import os
os.getcwd()
'''

# ---------------------------------------
### model building using dummy variables
# ----------------------------------------
bike.head()

bike2 = pd.read_csv(path)
bike2.head()

bike2.dtypes

_,fc=splitcols(bike2)
fc

bike2.season.unique()
bike2.mnth.unique() # convert the months to quarters
bike2.hr.unique() # split time in 6 hours interval
bike2.holiday.unique()
bike2.weekday.unique() # 0-3: weekday, 4-6: weekend
bike2.workingday.unique()
bike2.weathersit.unique() # 1:clear, 2:4 - not clear

# drop correlated column
bike2.drop(columns='atemp',inplace=True)
bike2.dtypes

# lets make the changes in the dataset
# 1) season: no change
bike2.season.unique()

# 2) mnth
bike2.mnth.unique()

quart = [ ['jan','feb','mar'], 
           ['apr','may','jun'],
           ['jul','aug','sep'],
           ['oct','nov','dec'] ]

newquart = ['q1','q2','q3','q4']

# create a new column and assign the mnth_quarter
bike2['mnth_quarter']=''

for i in range(len(quart)):
    bike2.mnth_quarter[bike2.mnth.isin(quart[i])] = newquart[i] 

bike2[['mnth','mnth_quarter']][bike2.mnth == 'dec'].head(3)
bike2.mnth_quarter.unique()

# delete the old column 'mnth'
bike2.drop(columns='mnth',inplace=True)

# rename the new column to the old value
bike2.rename(columns={'mnth_quarter':'mnth'},inplace=True)
bike2.mnth.unique()

# 3) hr : split time in 6 hours interval
bike2.hr.unique()
hrs = [range(0,6), range(6,12), range(12,18), range(18,24) ]
hrsval = ['earlymorn','morn','afternoon','evening']

# create a new column
bike2['hour'] = ''

for i in range(len(hrsval)):
    bike2.hour[bike2.hr.isin(hrs[i])] = hrsval[i]
    
bike2[['hr','hour']][bike2.hr.isin(hrs[3])]

# delete old column and rename the new col
bike2.drop(columns='hr',inplace=True)
bike2.rename(columns={'hour':'hr'},inplace=True)    
bike2.hr.unique()

# 4) holiday - no change
bike2.holiday.unique()

# 5) workingday - no change
bike2.workingday.unique()

# 6) weekday # 0-3: weekday, 4-6: weekend
bike2.weekday.unique()

# create a new column
bike2['wkday'] = 'weekday'
bike2.wkday[bike2.weekday > 3] = 'weekend'

bike2[['weekday','wkday']][bike2.weekday == 3]

# drop old column and rename new col
bike2.drop(columns='weekday',inplace=True)
bike2.rename(columns={'wkday':'weekday'},inplace=True)
bike2.weekday.unique()

# 7) weathersit
bike2.weathersit.unique() # 1:clear, 2:4 - not clear

# create a new column
bike2['w_sit'] = 'clear'
bike2.w_sit[bike2.weathersit > 1] = 'notclear'

bike2[['weathersit','w_sit']][bike2.weathersit == 1]

# drop old column and rename the new col
bike2.drop(columns='weathersit',inplace=True)
bike2.rename(columns={'w_sit':'weathersit'},inplace=True)
bike2.weathersit.unique()

# check the data types
bike2.dtypes

# convert the 'object' data types into dummy variables and build the next model
len(bike2.columns)
nc,fc = splitcols(bike2)

# make a copy of the dataset
bike2_copy = bike2.copy()

# bike2.season.unique()
# pd.get_dummies(bike2[fc[0]],drop_first=True).tail(20)

# create the dummy variables for the factor columns
for c in fc:
    dummy = pd.get_dummies(bike2[c],drop_first=True,prefix=c)
    bike2 = bike2.join(dummy)

print(bike2)

bike2.columns

# remove the original factor variables from the dataset
bike2.drop(columns=fc,inplace=True)

# check columns after deletion
bike2.columns


# build the next model on the dummy variables dataset
trainx3,testx3,trainy3,testy3 = splitdata(bike2,'target')
print(trainx3.shape, trainy3.shape, testx3.shape, testy3.shape)

# --------------------------------------- #
# build the linear regression model (M3)
# --------------------------------------- #
trainx3 = sm.add_constant(trainx3)
testx3 = sm.add_constant(testx3)

# Ordinary Least Square
m3 = sm.OLS(trainy3,trainx3).fit()

# summarise the model
m3.summary()

# predict 
p3 = m3.predict(testx3)
len(testy3)

# compare the Actual Y with the Predicted Y value
df3 = pd.DataFrame({'actual':testy3, 'predicted':np.round(p3,0)})
df3

# Mean Squared Error
mse3 = mean_squared_error(testy3,p3)
print("Model 1 \n\tmse={}, \n\trmse={}".format(round(mse1,2),round(np.sqrt(mse1),2)))
print("Model 3 \n\tmse={}, \n\trmse={}".format(round(mse3,2),round(np.sqrt(mse3),2)))

# from the results, it can be seen that the model with dummy variables is slightly better than first model

# build model 4 after removing the insignificant features, 1 at a time

# remove feature 'windspeed' 
bike4 = bike2.drop('windspeed',1)
bike4.columns

# build Model 4
trainx4,testx4,trainy4,testy4 = splitdata(bike4,'target')
print(trainx4.shape, trainy4.shape, testx4.shape, testy4.shape)

trainx4 = sm.add_constant(trainx4)
testx4 = sm.add_constant(testx4)

# Ordinary Least Square
m4 = sm.OLS(trainy4,trainx4).fit()

# summarise the model
m4.summary()

# predict 
p4 = m4.predict(testx4)

# Mean Squared Error
mse4 = mean_squared_error(testy4,p4)
print("Model 1 \n\tmse={}, \n\trmse={}".format(round(mse1,2),round(np.sqrt(mse1),2)))
print("Model 3 \n\tmse={}, \n\trmse={}".format(round(mse3,2),round(np.sqrt(mse3),2)))
print("Model 4 \n\tmse={}, \n\trmse={}".format(round(mse4,2),round(np.sqrt(mse4),2)))


### ----------------------------------------------------------------- ##


# ANOVA testing
# when features are categories and Y-continuous
# ensure that the categoeis are numeric (strings have to be converted to numbers for the Anova test to work)

# H0: means are same
# H1: means are not same

# check if 'season' is significant to predict bike sharing (target)

'''
model = ols('season~target',data=bike).fit()
anova = sm.stats.anova_lm(model,type=2)
# print(anova)
pvalue = anova['PR(>F)'][0]

if pvalue < 0.05:
    print('Reject H0: Feature is significant')
else:
    print('FTR H0: Feature is insignificant')
'''
  
def anovatest(x,y,data):
    model = ols('x~y',data=data).fit()
    anova = sm.stats.anova_lm(model,type=2)
    pvalue = anova['PR(>F)'][0]

    if pvalue < 0.05:
        msg = 'Reject H0: Feature {} is significant'.format(x.name)
    else:
        msg = 'FTR H0: Feature {} is insignificant'.format(x.name)
    
    return(msg)

anovatest(bike.mnth, bike.target,bike)
anovatest(bike.season,bike.target,bike)

## In-class proejct 
# log in LMS