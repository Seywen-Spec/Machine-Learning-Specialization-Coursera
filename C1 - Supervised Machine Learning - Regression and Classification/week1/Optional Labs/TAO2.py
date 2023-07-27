import math, copy
import numpy as np
import matplotlib.pyplot as plt

from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients


#Load our dataset
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

#Function to calculate cost function J
def compute_cost(x, y, theta, theta0):
    m = x.shape[0]
    cost = 0
    
    for i in range(m):
        h = theta * x[i] + theta0
        cost = cost + (h - y[i])**2
    total_cost = 1 / (2 * m) * cost
    
    return total_cost

def compute_gradient(x, y, theta, theta0):
    """Computes the gradient for linear regression

    Args:
        x (ndarray (m, )): Data, m examples
        y (ndarray (m, )): target values
        theta (scalar): leading coefficient
        theta0 (scalar): y intersection
    Returns:
        dj_dtheta (scalar) : The partial derivative of J with respect to theta
        dj_dtheta0 (scalar) : The partial derivative of J with respect to theta0
    N.B: Those should be calculated by hand, not computed
    """
    
    #Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        h = theta * x[i] + theta0
        dj_dtheta_i = (h - y[i]) * x[i]
        dj_dtheta0_i = (h - y[i])
        dj_dtheta += dj_dtheta_i
        dj_dtheta0 += dj_dtheta0_i
    dj_dtheta = dj_dtheta / m
    dj_dtheta0 = dj_dtheta0 / m
    
    return dj_dtheta, dj_dtheta0

#plt_gradients(x_train,y_train, compute_cost, compute_gradient)
#plt.show()


def gradient_descent(x, y, theta_in, theta0_in, alpha, num_iters, cost_function, gradient_function):
    """Performs gradient descent to fit theta, theta0. Update theta, theta0 by
    taking num_iters gradient steps with learning rate alpha.

    Args:
        x (ndarray (m, )): Data, m examples
        y (ndarray): target values
        theta_in (scalar): coef (initial value)
        theta0_in (scalar): y intersect (initial value)
        alpha (float): Learning rate
        num_iters (int): number of iterations to run gradient descent
        cost_function (func): funciton to call to produce cost
        gradient_function (func): function to call to produce gradient
    
    Returns:
        theta (scalar) : Updated value of parameter after running gradient descent
        theta0 (scalar) : Updated value of parameter after running gradient descent
        J_history (list) : History of cost function
        h_history (list) : History of parameters [theta, theta0]
    """
    
    theta = copy.deepcopy(theta_in) #avoids modifying global theta_in
    #An arrray to store cost J and theta's at each iteration primarly for graphing later
    J_history = []
    h_history = []
    theta0 = theta0_in
    theta = theta_in
    
    for i in range(num_iters):
        #Calculate the gradient and update the parameters
        dj_dtheta, dj_dtheta0 = gradient_function(x, y, theta, theta0)
        
        #Update Parameters using the equation above
        theta = theta - alpha * dj_dtheta
        theta0 = theta0 - alpha * dj_dtheta0
        
        #Save vost J at each iteration
        if i < 100000:
            J_history.append(cost_function(x, y, theta, theta0))
            h_history.append([theta, theta0])
        #Print every cost at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e}",
                  f"dj_dtheta : {dj_dtheta: 0.3e}, dj_dtheta0: {dj_dtheta0 : 0.3e}", 
                  f"theta: {theta: 0.3e}, theta0: {theta0: 0.5e}")
    
    return theta, theta0, J_history, h_history 

#initialize parameters 
theta_init = 0
theta0_init = 0
#some gradient descent settings
iterations = 10000
tmp_alpha =  1.0e-2
#run gradient descent 
theta_final, theta0_final, J_hist, h_hist = gradient_descent(x_train ,y_train, theta_init, theta0_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)

print(f"(w,b) found by gradient descent: ({theta_final:8.4f},{theta0_final:8.4f})") 

# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()

print(f"1000 sqft house prediction {theta_final*1.0 + theta0_init:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {theta_final*1.2 + theta0_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {theta_final*2.0 + theta0_final:0.1f} Thousand dollars")
    
fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, h_hist, ax)

fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, h_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5],
            contours=[1,5,10,20],resolution=0.5)
    
    
plt_divergence(h_hist, J_hist,x_train, y_train)
plt.show()
    
    
    