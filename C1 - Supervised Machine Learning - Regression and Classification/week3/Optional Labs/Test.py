import numpy as np
import matplotlib.pyplot as plt
from plt_overfit import overfit_example, output
from lab_utils_common import sigmoid
np.set_printoptions(precision=0.8)


def compute_cost_linear_reg(X, y, theta, theta0, lambda_ = 1):
    
    m = X.shape[0]
    n = len(theta)
    cost = 0.
    for i in range(m):
        h_i = np.dot(X[i], theta) + theta0
        cost = cost + (h_i - y[i])**2
    cost = cost / (2*m)
    
    reg_cost = 0
    for j in range(n):
        reg_cost += (theta[j]**2)
    reg_cost = (lambda_ / (2*m)) * reg_cost
    
    total_cost = reg_cost + cost
    
    return total_cost



def compute_cost_logistic_reg(X, y, theta, theta0, lambda_ = 1):
    
    m,n = X.shape
    cost = 0.
    for i in range(n):
        z_i = np.dot(X[i], theta) + theta0
        h_i = sigmoid(z_i)
        cost += -y[i]*np.log(h_i) - (1 - y[i])*np.log(1 - h_i)
    cost = cost / m
    
    reg_cost = 0
    for j in range(n):
        reg_cost += (theta[j]**2)
    reg_cost = (lambda_ / (2*m)) * reg_cost
    
    total_cost = cost + reg_cost
    
    return total_cost



def compute_gradient_linear_reg(X, y, theta, theta0, lambda_):
    
    m,n = X.shape
    dj_dtheta = np.zeros((n,))
    dj_dtheta0 = 0.
    
    for i in range(m):
        err = (np.dot(X[i], theta) + theta0) - y[i]
        for j in range(n):
            dj_dtheta[j]  = dj_dtheta[j] + err * X[i,j]
        dj_dtheta0 = dj_dtheta0 + err
    
    dj_dtheta = dj_dtheta / m
    dj_dtheta0 = dj_dtheta0 / m
    
    for j in range(m):
        dj_dtheta[j] = dj_dtheta[j] + (lambda_/m) * theta
    
    return dj_dtheta, dj_dtheta0



def compute_gradient_logistic_reg(X, y, theta, theta0, lambda_):
    
    m,n = X.shape
    dj_dtheta = np.zeros((n,))
    dj_dtheta0 = 0.
    
    for i in range(m):
        z_i = (np.dot(X[i], theta) + theta0)
        h_i = sigmoid(z_i)
        err_i = h_i - y[i]
        for j in range(n):
            dj_dtheta[j] = dj_dtheta[j] + err_i * X[i,j]
        dj_dtheta0 = dj_dtheta0 + err_i
    dj_dtheta0 = dj_dtheta0 / m
    dj_dtheta = dj_dtheta / m
    
    for j in range(n):
        dj_dtheta[j] = dj_dtheta[j] + (lambda_/m) * theta
    
    return dj_dtheta, dj_dtheta0