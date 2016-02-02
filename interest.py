
# coding: utf-8

import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from math import sqrt


train = pd.read_csv('C:\\Users\\Nj_neeraj\\Documents\Data_Science\\sample_work_state_farm\\StateFarmDataScienceWORKSAMPLE\\Data_for_Cleaning&_Modeling.csv',
                   sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
test = pd.read_csv('C:\\Users\\Nj_neeraj\\Documents\Data_Science\\sample_work_state_farm\\StateFarmDataScienceWORKSAMPLE\\Holdout_for_Testing.csv',
                   sep=',', error_bad_lines=False, index_col=False, dtype='unicode')


#reading first 10 data values.
train.head(n=10)

#to see total number of rows.
len(train)
print(train.dtypes)

#summarizing the data
train.describe()

#finding no. of missing values.
len(train.index)-train.count()
#train_X.isnull().sum()

#creating a copy.
train_X = train.copy()
test_X = test.copy()

#filling missing values.
train_X = train_X.fillna(-999)
test_X = test_X.fillna(-999)


#cleaning a bit for train.
train_X[['X1', 'X30','X21']] = train_X[['X1', 'X30','X21']].replace('%','',regex=True).astype('float')
train_X[['X4', 'X5','X6']] = train_X[['X4', 'X5','X6']].replace(to_replace=',', value='', regex=True)
train_X['X7'] = train_X[['X7']].replace(to_replace='months', value='', regex=True)

#convert number of months to years for better readability.
train_X['X7'] = train_X['X7'].astype('float') / 12

##cleaning a bit for test.
test_X[['X1', 'X30','X21']] = test_X[['X1', 'X30','X21']].replace('%','',regex=True).astype('float')
test_X[['X4', 'X5','X6']] = test_X[['X4', 'X5','X6']].replace(to_replace=',', value='', regex=True)
test_X['X7'] = test_X[['X7']].replace(to_replace='months', value='', regex=True)

#convert number of months to years for better readability.
test_X['X7'] = test_X['X7'].astype('float') / 12

print(train_X.dtypes)

train_X.head(n=5)

# getting result variable.
train_y = np.array(train["X1"])


## categical column name list ##
categorical_columns = ["X7","X12","X14","X17","X24",'X32']
#encoding categorical variable
for var in categorical_columns:
    lb = preprocessing.LabelEncoder()
    full_var_data = pd.concat((train_X[var],test_X[var]),axis=0).astype('str')
    lb.fit( full_var_data )
    train_X[var] = lb.transform(train_X[var].astype('str'))
    test_X[var] = lb.transform(test_X[var].astype('str'))


## Dropping the unnecessary column ##
train_X = train_X.drop(['X1'],axis=1) 
print ("Train shape is : ",train_X.shape)
print ("Test shape is : ",test_X.shape)

# converting row values to a form where it can be dealt with
train_X.loc[train_X.X11 == '< 1 year', 'X11'] = 0
train_X.loc[train_X.X11 == 'n/a', 'X11'] = 0
train_X.loc[train_X.X11 == '1 year', 'X11'] = 1
train_X.loc[train_X.X11 == '2 years', 'X11'] = 2
train_X.loc[train_X.X11 == '3 years', 'X11'] = 3
train_X.loc[train_X.X11 == '4 years', 'X11'] = 4
train_X.loc[train_X.X11 == '5 years', 'X11'] = 5
train_X.loc[train_X.X11 == '6 years', 'X11'] = 6
train_X.loc[train_X.X11 == '7 years', 'X11'] = 7
train_X.loc[train_X.X11 == '8 years', 'X11'] = 8
train_X.loc[train_X.X11 == '9 years', 'X11'] = 9
train_X.loc[train_X.X11 == '10+ years', 'X11'] = 10

test_X.loc[test_X.X11 == '< 1 year', 'X11'] = 0
test_X.loc[test_X.X11 == 'n/a', 'X11'] = 0
test_X.loc[test_X.X11 == '1 year', 'X11'] = 1
test_X.loc[test_X.X11 == '2 years', 'X11'] = 2
test_X.loc[test_X.X11 == '3 years', 'X11'] = 3
test_X.loc[test_X.X11 == '4 years', 'X11'] = 4
test_X.loc[test_X.X11 == '5 years', 'X11'] = 5
test_X.loc[test_X.X11 == '6 years', 'X11'] = 6
test_X.loc[test_X.X11 == '7 years', 'X11'] = 7
test_X.loc[test_X.X11 == '8 years', 'X11'] = 8
test_X.loc[test_X.X11 == '9 years', 'X11'] = 9
test_X.loc[test_X.X11 == '10+ years', 'X11'] = 10

##removing rows with Number of years employed less than 1, we will consider them as outliers.
train_X = train_X[(train_X.X11 >= 1)]

len(train_X)

##extracting month and day from date loan was issued, since year is same as 2016
train_X['day'] = train_X['X15'].str.extract('(-..)')
train_X['day'] = train_X['day'].replace('-','',regex=True)
#train_X['day']
train_X['monthloan'] = train_X['X15'].str.extract('(...)')
train_X['monthloan']
test_X['day'] = test_X['X15'].str.extract('(..-)')
test_X['day'] = test_X['day'].replace('-','',regex=True)
#test_X['day']
test_X['monthloan'] = test_X['X15'].str.extract('(-...)')
test_X['monthloan'] = test_X['monthloan'].replace('-','',regex=True)
#test_X['monthloan']


#####
## same for credit line date.
train_X['credityear'] = train_X['X23'].str.extract('(-..)')
train_X['credityear'] = train_X['credityear'].replace('-','',regex=True)
train_X['monthcredit'] = train_X['X23'].str.extract('(...)')
#train_X['monthcredit']


test_X['credityear'] = test_X['X23'].str.extract('(..-)')
test_X['credityear'] = test_X['credityear'].replace('-','',regex=True)
#test_X['day']
test_X['monthcredit'] = test_X['X23'].str.extract('(-...)')
test_X['monthcredit'] = test_X['monthcredit'].replace('-','',regex=True)
