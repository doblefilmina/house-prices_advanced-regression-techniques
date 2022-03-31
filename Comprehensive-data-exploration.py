## imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
#warnings.filterwarnings('ignore')

#%matplotlib inline

## set working directory
os.chdir("c:/Users/New/Documents/kaggle/house-prices_advanced-regression-techniques")

## load data
df_train = pd.read_csv("train.csv")

## check the data
df_train.columns

## analyse SalePrice
df_train['SalePrice'].describe()

##histogram
sns.distplot(df_train['SalePrice'])

##boxplot
sns.boxplot(df_train['SalePrice'])

##skewness and kurtosis
print('Skewness: %f' % df_train['SalePrice'].skew())
print('Kurtosis: %f' % df_train['SalePrice'].kurt())

####Relationship with numerical variables
##Scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0 , 800000))

##Scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

####Relationship with categorical features
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)

#box plot YearBuilt/SalePrice
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(20,8))
fig = sns.boxplot(x=var, y='SalePrice', data = data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

####Correlation matrix (heatmap style)
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.8, square=True)

#Saleprice correlation matrix (zoomed heatmap style)
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='2f', annot_kws ={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Scatterplot
sns.set()
cols=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()

#### Missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count() ).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#### I`m following a tutorial, so I will follow it. But I don't agree with the following
##deleting missing data variables
df_train_drop = df_train.drop((missing_data[missing_data['Total'] > 1 ]).index, 1)
df_train_drop = df_train_drop.drop(df_train_drop.loc[df_train_drop['Electrical'].isnull()].index)
print(df_train_drop.isnull().sum().max()) #just checking that there's no missing data missing...

#### Outliers

##Standardize data
saleprice_scaled = StandardScaler().fit_transform(df_train_drop['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('outer range (high) of the distribution:')
print(high_range)

##At the moment we won't consider any of the values as outliers, but the biggest are suspicious.
## Also keep in mind that the distribution has positive skewness

#### Bivariate analysis
var = 'GrLivArea'
data = pd.concat([df_train_drop['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

#Deleting points
df_train_drop.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train_drop = df_train_drop.drop(df_train_drop[df_train_drop['Id'] == 1299].index)
df_train_drop = df_train_drop.drop(df_train_drop[df_train_drop['Id'] == 524].index)
 
# Bivariate analysis saleprice/TotalBsmtSF
var = 'TotalBsmtSF'
data = pd.concat([df_train_drop['SalePrice'], df_train_drop[var]], axis = 1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# Verify normality
#histogram and normal probability plot
sns.distplot(df_train_drop['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train_drop['SalePrice'], plot=plt)

##Not normal, so apply log transformation
df_train_drop['SalePrice'] = np.log(df_train_drop['SalePrice'])

#transformed histogram and normal probability plot
sns.distplot(df_train_drop['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.problpot(df_train_drop['SalePrice'], plot=plt)

#verify normality in GrLivArea
sns.distplot(df_train_drop['GrLivArea'], fit=norm)
fig = plt.figure ()
res = stats.probplot(df_train_drop['GrLivArea'], plot=plt)

#Looks as it has skewness
#Apply log transformation
df_train_drop['GrLivArea'] = np.log(df_train_drop['GrLivArea'])

#transformed histogram and normal probability plot
sns.distplot(df_train_drop['GrLivArea'])
fig = plt.figure()
res = stats.probplot(df_train_drop['GrLivArea'], plot=plt)

#verify normality in TotalBsmtSF
sns.distplot(df_train_drop['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train_drop['TotalBsmtSF'], plot=plt)

#Looks as it has skewness
#There are some values equal to zero that they can't be applyed logaritm
#Create a new variable, categorical, has or has not basement.

df_train_drop['hasBsmt'] = pd.Series(len(df_train_drop['TotalBsmtSF']), index=df_train_drop.index)
df_train_drop['hasBsmt'] = 0
df_train_drop.loc[df_train_drop['TotalBsmtSF']>0, 'hasBsmt'] = 1

#Apply log transformation
df_train_drop.loc[df_train_drop['hasBsmt']==1, 'TotalBsmtSF'] = np.log(df_train_drop['TotalBsmtSF'])

#verify normality
sns.distplot(df_train_drop[df_train_drop['hasBsmt']==1]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train_drop[df_train_drop['hasBsmt']==1]['TotalBsmtSF'], plot=plt)

#### Look for homoscedasticity
#scatter plot of SalePrice vs GrLivArea
plt.scatter(df_train_drop['GrLivArea'], df_train_drop['SalePrice'])

plt.scatter(df_train_drop[df_train_drop['hasBsmt']==1]['TotalBsmtSF'], df_train_drop[df_train_drop['hasBsmt']==1]['SalePrice'])

###Convert categorical variable into dummy
df_train_drop = pd.get_dummies(df_train_drop)