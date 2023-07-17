#Importing libraries and reading CSV Files
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df1 = pd.read_csv("https://raw.githubusercontent.com/imonishkumar/AI-Projects/main/mental%20and%20substance%20use%20as%20share%20of%20disease.csv")
df2=pd.read_csv("https://raw.githubusercontent.com/imonishkumar/AI-Projects/main/prevalence%20by%20mental%20and%20substance%20use%20disorder.csv")
data = pd.merge(df1, df2)
data.isnull().sum()
data.drop('Code',axis=1,inplace=True)
data.size,data.shape
data.set_axis(['Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol','mental_fitness'], axis='columns', inplace=True)
mean = data['mental_fitness'].mean()
df = data.copy()

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in df.columns:
    if df[i].dtype == 'object':
        df[i]=l.fit_transform(df[i])

X = df.drop('mental_fitness',axis=1)
y = df['mental_fitness']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)

X = df.drop('mental_fitness',axis=1)
y = df['mental_fitness']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Define the regression algorithms to be evaluated
regressors = [
    ('Linear Regression', LinearRegression()),
    ('Ridge', Ridge()),
    ('Lasso', Lasso()),
    ('SVR', SVR()),
    ('Random Forest', RandomForestRegressor())
]

# Define the parameter grid for each regressor
param_grids = {
    'Linear Regression': {},
    'Ridge': {'alpha': [0.01, 0.1, 1.0]},
    'Lasso': {'alpha': [0.01, 0.1, 1.0]},
    'SVR': {'C': [0.01, 0.1, 1.0], 'kernel': ['linear', 'rbf']},
    'Random Forest': {'n_estimators': [10, 50, 100]}
}

# Perform grid search to find the best regressor
best_regressor = None
best_rmse = np.inf

for name, regressor in regressors:
    grid_search = GridSearchCV(regressor, param_grid=param_grids[name], scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(xtrain, ytrain)
    ytrain_pred = grid_search.predict(xtrain)
    rmse = np.sqrt(mean_squared_error(ytrain, ytrain_pred))
    if rmse < best_rmse:
        best_rmse = rmse
        best_regressor = grid_search.best_estimator_

# Evaluate the best regressor on the training set
ytrain_pred = best_regressor.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = np.sqrt(mean_squared_error(ytrain, ytrain_pred))
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for the training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# Evaluate the best regressor on the testing set
ytest_pred = best_regressor.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = np.sqrt(mean_squared_error(ytest, ytest_pred))
r2 = r2_score(ytest, ytest_pred)

print("The model performance for the testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
