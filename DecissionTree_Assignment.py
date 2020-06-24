# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:06:48 2020

@author: RADHIKA
"""
###############Company data Assignment##################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
company_data=pd.read_csv("D:\\ExcelR Data\\Assignments\\Decission Trees\\Company_Data.csv",encoding = "ISO-8859-1")
company_data.columns
company_data.head()
company_data.shape
company_data.describe()
company_data.hist()
company_data.isnull().sum()

company_data['Urban'],Urban = pd.factorize(company_data['Urban'])
company_data['US'],US = pd.factorize(company_data['US'])
company_data.columns
company_data.head()
##Converting the sales  variable to bucketing. 
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in company_data.columns:
    if company_data[column_name].dtype == object:
        company_data[column_name] = le.fit_transform(company_data[column_name])
    else:
        pass
X=company_data.drop(['Sales'],axis=1)
X
company_data['Sales_levels']=np.where(company_data['Sales']>=7.5,'high','low')
y=company_data['Sales_levels']
y

y.value_counts()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#Accuracy
from sklearn import metrics
DT = metrics.accuracy_score(y_test, y_pred) * 100
print("\nThe accuracy score using the DecisionTreeClassifier : ",DT)
#####Accuracy is 100%


######################Fraud check Assignment#################

import pandas as pd
import numpy as np
fraud= pd.read_csv("D:\\ExcelR Data\\Assignments\\Decission Trees\\Fraud_check.csv")
fraud.columns
fraud.describe()
fraud.shape
fraud.isnull().sum
##Converting the Taxable income variable to bucketing. 
fraud["income"]="<=30000"
fraud.loc[fraud["Taxable.Income"]>=30000,"income"]="Good"
fraud.loc[fraud["Taxable.Income"]<=30000,"income"]="Risky"
fraud.drop(["Taxable.Income"],axis=1,inplace=True)
fraud.columns
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in fraud.columns:
    if fraud[column_name].dtype == object:
        fraud[column_name] = le.fit_transform(fraud[column_name])
    else:
        pass
  
##Splitting the data into featuers and labels
features = fraud.iloc[:,0:5]
features
labels = fraud.iloc[:,5]
labels
## Collecting the column names
colnames = list(fraud.columns)
predictors = colnames[0:5]
target = colnames[5]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)
from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn import metrics
DT = metrics.accuracy_score(y_test, y_pred) * 100
print("\nThe accuracy score using the DecisionTreeClassifier : ",DT)
###The accuracy score using the DecisionTreeClassifier :  0.8333333333333334
#83%