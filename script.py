import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 

df = pd.read_csv('50_Startups.csv') 

heading = ['R&D Spend','Administration','Marketing Spend','Profit']

df_core = df[heading]

x_name = "Administration"
y_name = "Profit"

# TODO: Find a feature best fitting to a Profit 
sns.lmplot(x =x_name, y=y_name, data = df_core, order = 2, ci = None) 
plt.show()

# TODO: use more than a feature to train your model
X = np.array(df_core[x_name]).reshape(-1, 1) 
y = np.array(df_core[y_name]).reshape(-1, 1) 
  
# Separating the data into independent and dependent variables 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 

# Splitting the data into training and testing data 
regr = LinearRegression() 

# TODO: run this program a few times, why the result change overtime
regr.fit(X_train, y_train) 
print(regr.score(X_test, y_test)) 


y_pred = regr.predict(X_test) 
plt.scatter(X_test, y_test, color ='b') 
plt.plot(X_test, y_pred, color ='k') 

# Data scatter of predicted values 
plt.show() 

