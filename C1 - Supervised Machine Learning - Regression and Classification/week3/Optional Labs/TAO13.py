import numpy as np
import matplotlib.pyplot as plt
from plt_overfit import overfit_example, output
from lab_utils_common import sigmoid
np.set_printoptions(precision=0.8)


def compute_cost_linear_reg(X, y, theta, theta0, lamda_ = 1):
    """Computes the cost over all examples

    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m, )): target values
        theta (ndarray (n,)): model parameters
        theta0 (scalar): model parameter
        lamda_ (scalar): Control amount of regulization. Defaults to 1.
        
    Returns:
        total_cost (scalar) = cost
    """

    m = X.shape[0]
    n = len(theta)
    cost = 0.
    for i in range(m):
        h_i = np.dot(X[i], theta) + theta0
        cost = cost + (h_i - y[i]) ** 2
    cost = cost / (2*m)

    reg_cost = 0
    for j in range(n):
        reg_cost += (theta[j]**2)
    reg_cost = (lamda_/(2*m)) * reg_cost
    
    total_cost = cost + reg_cost
    return total_cost

#Data Implimentation
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
theta_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
theta0_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, theta_tmp, theta0_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)



def compute_cost_logistic_reg(X, y, theta, theta0, lambda_ = 1):
    """Computes the vost over all examples

    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m, )): target values
        theta (ndarray (n, )): model parameters
        theta0 (scalar): model parameter
        lambda_ (scalar): Controls the amount of regularization. Defaults to 1.
        
    Returns:
        total_cost (scalar) : cost
    """
    
    m,n = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], theta) + theta0
        h_i = sigmoid(z_i)
        cost += -y[i]*np.log(h_i) - (1-y[i])*np.log(1 - h_i)    
    
    cost = cost / m
    
    reg_cost = 0
    for j in range(n):
        reg_cost += (theta[j]**2)
    reg_cost = (lambda_ / (2*m)) * reg_cost
    
    total_cost = cost + reg_cost
    return total_cost

#Data Implimentation
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)    



def compute_gradient_linear_reg(X, y, theta, theta0, lambda_):
    """Computes the gradient for linear regression

    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m, )): target values
        theta (ndarray (n, )): model parameters
        theta0 (scalar): model parameter
        lambda_ (scalar): Controls the amount of regularization
        
    Returns:
        dj_dtheta (ndarray (n, )): The partial derivative of the cost funciton with respect to theta
        dj_dtheta0 (scalar)      : The partial derivative of the cost function with respect to theta0
    """
    
    m,n = X.shape
    dj_dtheta = np.zeros((n,))
    dj_dtheta0 = 0.
    
    for i in range(m):
        err = (np.dot(X[i], theta) + theta0) - y[i]
        for j in range(n):
            dj_dtheta[j] = dj_dtheta[j] + err * X[i, j]
        dj_dtheta0 = dj_dtheta0 + err
    dj_dtheta = dj_dtheta / m
    dj_dtheta0 = dj_dtheta0 / m
    
    for j in range(m):
        dj_dtheta[j] = dj_dtheta[j] + (lambda_/m) * theta[j]
        
    return dj_dtheta, dj_dtheta0


#Data Implimentation
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )



def compute_gradient_logistic_reg(X, y, theta, theta0, lambda_):
    """Computes the gradient for logistic regression

    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        theta (ndarray (n, )): model parameters
        theta0 (scalar): model parameter
        lambda_ (scalar): Controls amount of regularization
        
    Returns:
        dj_dtheta (ndarray (n, )) : The partial derivative of the cost funciton with respect to theta
        dj_dtheta0 (scalar)       : The partial derivative of the cost function with respect to theta0
    """
    
    m,n = X.shape
    dj_dtheta = np.zeros((n,))
    dj_dtheta0 = 0.0
    
    for i in range(m):
        h_i = sigmoid(np.dot(X[i], theta) + theta0)
        err_i = h_i - y[i]
        for j in range(n):
            dj_dtheta[j] = dj_dtheta[j] + err_i * X[i,j]
        dj_dtheta0 = dj_dtheta0 + err_i
    dj_dtheta = dj_dtheta / m
    dj_dtheta0 = dj_dtheta0 / m
    
    for j in range(n):
        dj_dtheta[j] = dj_dtheta[j] + (lambda_/m) * theta[j]
        
    return dj_dtheta, dj_dtheta0


#Data Implimentation
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )




    