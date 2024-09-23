import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score 

#Load the dataset
df = pd.read_csv("Housing.csv")


#Display the first 5 rows of the dataset
print(df.head())


#Display statistics
print(df.describe())


#Check for missing values
print(df.isnull().sum())


#Data Cleaning
df.dropna(inplace=True)
df.info()

#EDA
df.hist(figsize=(20,10))
plt.figure(figsize = (10, 5))
sns.boxplot(x = 'furnishingstatus', y = 'price', hue = 'airconditioning', data = df)
plt.show()

#Preparation of Data
varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
def binary_map(x):
    return x.map({'yes': 1, "no": 0})
df[varlist] = df[varlist].apply(binary_map)
df[varlist].head()

df.head()
status = pd.get_dummies(df['furnishingstatus'], dtype=int)
status.head()
status = pd.get_dummies(df['furnishingstatus'], dtype=int, drop_first = True)
df = pd.concat([df, status], axis = 1)
df.head()

df.drop(['furnishingstatus'], axis = 1, inplace = True)
df.head()
df.corr()
plt.figure(figsize = (15,8))
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show()


#Spitting data into Testing & Trainning Data Separately
np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.8, test_size = 0.2, random_state = 100)

scaler = MinMaxScaler()
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()

df_train.describe()
plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()

y_train = df_train.pop('price')
X_train = df_train

#Model Building
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, step=6)
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))

col = X_train.columns[rfe.support_]
col

#Building model using statsmodel, for the detailed statistics

X_train_rfe = X_train[col]
X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()
print(lm.summary())


#Calculate the VIFs for the model

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


#Residual Analysis of the train data
y_train_price = lm.predict(X_train_rfe)
res = (y_train_price - y_train)
fig = plt.figure()
sns.histplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  
plt.xlabel('Errors', fontsize = 18) 

plt.scatter(y_train,res)
plt.show()

#Model Evaluation
num_vars = ['area','stories', 'bathrooms', 'airconditioning', 'prefarea','parking','price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])

#Dividing into X_test and y_test
y_test = df_test.pop('price')
X_test = df_test
X_test = sm.add_constant(X_test)
X_test_rfe = X_test[X_train_rfe.columns]
y_pred = lm.predict(X_test_rfe)
r2_score(y_test, y_pred)

fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)               
plt.xlabel('y_test', fontsize=18)                         
plt.ylabel('y_pred', fontsize=16)   