# Mental-Fitness-Tracker
Mental Health Fitness-Tracker
The Mental Health Fitness Tracker project focuses on analyzing and predicting mental fitness levels of individuals from various countries with different mental disorders. It utilizes regression techniques to provide insights into mental health and make predictions based on the available data. The project also provides a platform for users to track their mental health and fitness levels. The project is built using Python.

Table of Contents
Mental Health Fitness-Tracker
Table of Contents
Installation
Usage
Contributing
License
Installation
To use the code and run the examples, follow these steps:

Ensure that you have Python 3.x installed on your system.
Install the required libraries by running the following command:
pip install pandas numpy matplotlib seaborn scikit-learn plotly.express
Download the project files and navigate to the project directory.
Usage
Select the country and the mental disorder you want to analyze.
Select the year range you want to analyze.
Click on the "Analyze" button.
The app will display the results of the analysis.
Click on the "Predict" button to predict the mental fitness level of the selected country.
The app will display the predicted mental fitness level of the selected country.
Click on the "Track" button to track your mental fitness level.
The app will display the results of the tracking.
Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the repo
Clone the project
Create your feature branch
Commit your changes
Push to the branch
Open a pull request
License
Distributed under the ICS License.

References
Datasets that were useD in here were taken from KAGGLE
This project was made during my internship period for Edunet Foundation in association with IBM SkillsBuild and AICTE
PYTHON CODE
IMPORTING REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
READING DATASETS
df1 = pd.read_csv('mental-and-substance-use-as-share-of-disease.csv')
df2=pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")
SHOW DATA SET
df1.head()
df2.head()
MERGING TWO DATASETS
data = pd.merge(df1, df2)
data.head()
DATA CLEANING
data.isnull().sum()
data.drop('Code', axis=1, inplace=True)
data.size,data.shape
RENAMED COLUMNS
data.set_axis(['Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol','mental_fitness'], axis='columns', inplace=True)
data.head()
EXPLORATORY ANALYSIS
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True,cmap='Greens')
plt.plot()
sns.jointplot(data,x="Schizophrenia",y="mental_fitness",kind="reg",color="m")
plt.show()
sns.jointplot(data,x='Bipolar_disorder',y='mental_fitness',kind='reg',color='blue')
plt.show()
sns.pairplot(data,corner=True)
plt.show()
mean = data['mental_fitness'].mean()
mean
fig = px.pie(data, values='mental_fitness', names='Year')
fig.show()
fig=px.bar(data.head(10),x='Year',y='mental_fitness',color='Year',template='ggplot2')
fig.show()
YEARWISE VARIATIONS IN MENTAL FITNESS OF DIFFERENT COUNTRIES
fig = px.line(data, x="Year", y="mental_fitness", color='Country',markers=True,color_discrete_sequence=['red','blue'],template='plotly_dark')
fig.show()
df=data.copy()
df.head()
df.info()
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
print("xtrain: ", xtrain.shape)
print("xtest: ", xtest.shape)
print("ytrain: ", ytrain.shape)
print("ytest: ", ytest.shape)
LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()
lr.fit(xtrain,ytrain)

# model evaluation for training set
ytrain_pred = lr.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = lr.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
RANDOM FOREST REGRESSOR
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(xtrain, ytrain)

# model evaluation for training set
ytrain_pred = rf.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = rf.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
