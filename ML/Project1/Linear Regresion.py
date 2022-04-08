#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
#----------------------------------------------------
#Read dataset
dataset = pd.read_csv("Salary_Data.csv")

#X Data
X = dataset.iloc[:, [0]].values

#y Data
y = dataset.iloc[:, 1].values
#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0, shuffle =True)
#----------------------------------------------------
#Applying Linear Regression Model 

LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)
LinearRegressionModel.fit(X_train, y_train)

#Calculating Prediction
y_pred = LinearRegressionModel.predict(X_test)
print('Predicted Value for Linear Regression is : ' , y_pred[:10])


#----------------------------------------------------
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)

#----------------------------------------------------
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)

#----------------------------------------------------
#Calculating Median Squared Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )

#----------------------------------------------------
#Visualizaton Training set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, LinearRegressionModel.predict(X_train), color = 'blue')
plt.title('Salart vs Experience (Training set)')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#----------------------------------------------------
#Visualizaton Test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, LinearRegressionModel.predict(X_train), color = 'blue')
plt.title('Salart vs Experience (Test set)')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
