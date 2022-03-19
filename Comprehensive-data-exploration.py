## imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline

## set working directory
os.chdir("c:/Users/New/Documents/kaggle/house-prices_advanced-regression-techniques")

## load data
df_train = pd.read_csv("train.csv")

## check the data
df_train.columns

## analyse SalePrice
df_train['SalePrice'].describe()



