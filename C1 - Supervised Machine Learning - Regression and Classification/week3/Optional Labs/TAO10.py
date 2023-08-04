import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import plot_data, sigmoid, dlc

#Dataset
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)

fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()

def compute_cost_logistic(X, y, theta, theta0):
    """Computes cost

    Args:
        X (ndarray (m,n )): Data, m examples with n features
        y (ndarray (m, )): target values
        theta (ndarray (n, )): model parameters
        theta0 (scalar): model parameter
        
    Returns:
        cost (scalar) : cost
    """
    
    m = X.shape[0]
    cost = 0.0
    
    for i in range(m):
        z_i = np.dot(X[i], theta) + theta0
        h_i = sigmoid(z_i)
        cost += -y[i]*np.log(h_i) - (1 - y[i]) * (np.log(1- h_i))
        
    cost = cost / m
    return cost

w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))


