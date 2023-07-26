import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl


x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])           #(price in 1000s of dollars)

def compute_cost(x, y, theta, theta_zero):
    """Computes the corst funciton for linear regression

    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target values
        theta (scalar): leading coefficient
        theta_zero (scalar): y intersection
    """
    #number of training examples
    m = x.shape[0]
    
    cost_sum = 0
    for i in range(m):
        h = theta * x[i] + theta_zero
        cost = (h - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1/ (2*m)) * cost_sum
    
    return total_cost

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])


fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

soup_bowl()