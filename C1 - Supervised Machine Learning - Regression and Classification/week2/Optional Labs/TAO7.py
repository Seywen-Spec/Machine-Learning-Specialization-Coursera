import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from lab_utils_multi import load_house_data
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)


X_train = np.array([1.0, 2.0])   #features
y_train = np.array([300, 500])   #target value

linear_model = LinearRegression()
#X must be a 2-D Matrix
linear_model.fit(X_train.reshape(-1, 1), y_train) 

theta = linear_model.coef_
theta0 = linear_model.intercept_
print(f"theta = {theta:}, theta0 = {theta0:0.2f}")
print(f"'manual' prediction: h = theta * x + thetha0 : {1200*theta + theta0}")


y_pred = linear_model.predict(X_train.reshape(-1, 1))

print("Prediciton on training set:", y_pred)

X_test = np.array([[1200]])
print(f'Prediciotn for 1200 sqft house: ${linear_model.predict(X_test)[0]:0.2f}')



############Second Example
# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

linear_model = LinearRegression()
linear_model.fit(X_train, y_train) 

theta0 = linear_model.intercept_
theta = linear_model.coef_
print(f"theta = {theta:}, theta0 = {theta0:0.2f}")

print(f"Prediction on training set:\n {linear_model.predict(X_train)[:4]}" )
print(f"prediction using w,b:\n {(X_train @ theta + theta0)[:4]}")
print(f"Target values \n {y_train[:4]}")

x_house = np.array([1200, 3,1, 40]).reshape(-1,4)
x_house_predict = linear_model.predict(x_house)[0]
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.2f}")