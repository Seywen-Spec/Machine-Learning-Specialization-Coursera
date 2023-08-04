import math, copy
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
from plt_quad_logistic import plt_quad_logistic, plt_prob


#Dataset
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()


def compute_gradient_logistic(X, y, theta, theta0):
    """Computes the gradient for logistic regression

    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m, )): target values
        theta (ndarray (n, )): model parameters
        theta0 (scalar): model parameter
        
    Returns:
        dj_dtheta (ndarray (n,)) : The partial derivative of the cost function with respect to theta
        dj_dtheta0 (scalar) : The partial derivative of the cost funciton with respect to theta0
    """
    
    m,n = X.shape
    dj_dtheta = np.zeros((n,))
    dj_dtheta0 = 0.0
    
    for i in range(m):
        h_i = sigmoid(np.dot(X[i], theta) + theta0)
        err_i = h_i - y[i]
        for j in range(n):
            dj_dtheta[j] = dj_dtheta[j] + err_i *X[i, j]
        dj_dtheta0 = dj_dtheta0 + err_i
        
    dj_dtheta = dj_dtheta / m
    dj_dtheta0 / dj_dtheta0 / m
    
    return dj_dtheta, dj_dtheta0

X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )



def gradient_descent(X, y, theta_in, theta0_in, alpha, num_iters):
    """Performs batch gradient descent

    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m, )): target values
        theta_in (ndarray(n, )): Initial values of model parameters
        theta0_in (scalar): Initial valu of model parameter
        alpha (float): Learning rate
        num_iters (_type_): number of iterations to run gradient descent 
        
    Returns:
        theta (ndarray (n,)) : Updated calues of parameters
        theta0 (scalar)      : Updated value of parameter
    """
    
    #An array to store cost J and theta's at each iteration primarily for graphing later
    J_history = []
    theta = copy.deepcopy(theta_in)
    theta0 = theta0_in
    
    for i in range(num_iters):
        #Caclulate the gradient and update the parameters
        dj_dtheta, dj_dtheta0 = compute_gradient_logistic(X, y, theta, theta0)
        
        #Update Parameters using theta, theta0, alpha and gradient
        theta = theta - alpha * dj_dtheta
        theta0 = theta0 - alpha * dj_dtheta0
        
        #Save cost J at each iteration
        if i < 100000:          #prevent resrouce exhaustion
            J_history.append( compute_cost_logistic(X, y, theta, theta0))
            
        #Print cost every at intervals 10 times or as many iterations if < 10
        if i& math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}")
            
    return theta, theta0, J_history

theta_tmp  = np.zeros_like(X_train[0])
theta0_tmp  = 0.
alph = 0.1
iters = 10000

theta_out, theta0_out, _ = gradient_descent(X_train, y_train, theta_tmp, theta0_tmp, alph, iters) 
print(f"\nupdated parameters: w:{theta_out}, b:{theta0_out}")
    

fig,ax = plt.subplots(1,1,figsize=(5,4))
# plot the probability 
plt_prob(ax, theta_out, theta0_out)

# Plot the original data
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')   
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)

# Plot the decision boundary
x0 = -theta_out/theta_out[0]
x1 = -theta0_out/theta0_out[1]
ax.plot([0,x0],[x1,0], c=dlc["dlblue"], lw=1)
plt.show()
            
