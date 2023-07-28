import copy, math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2) # reduced display precision on numpy arrays

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# data is stored in numpy array/matrix
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)


theta0_init = 785.1811367994083
theta_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"theta_init shape: {theta_init.shape}, theta0_init type: {type(theta0_init)}")


#Single Prediction element by element
def predict_single_loop(x, theta, theta0):
    """Single predict using linear regression

    Args:
        x (ndarray): shape (n, ) example with multiple features
        theta (ndarray): shape (n, ) model parameters (leading coefs)
        theta0 (scalar): model paramter ('y' intersect)
    
    Returns:
        h (scalar): hypothesis
    """
    n = x.shape[0]
    h = 0
    for i in range(n):
        h_i = theta[i] * x[i]
        h = h + h_i
    h = h + theta0
    return h

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
h = predict_single_loop(x_vec, theta_init, theta0_init)
print(f"h shape {h.shape}, prediction: {h}")


#Single Prediction, vector
def predict(x, theta, theta0):
    """single predict using linear regression

    Args:
        x (ndaray): shape (n, ) example with multiple features
        theta (ndarray): shape (n, ) model parameter (leading coefs)
        theta0 (scalar): model parameter ('y' intersection)
    Returns:
    h (scalar): hypothesis
    """
    h = np.dot(x, theta) + theta0
    return h

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
h = predict(x_vec, theta_init, theta0_init)
print(f"h shape {h.shape}, prediction: {h}")


def compute_cost(X, y, theta, theta0):
    """compute cost

    Args:
        X (ndarray (m, n)): Data,  m examples with n features
        y (ndarray (m, )): target values
        theta (ndarray (n, )): model parameter
        theta0 (scalar): model parameter
        
    Returns:
        cost (scalar): cost
    """
    
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        h_i = np.dot(X[i], theta) + theta0
        cost = cost + (h_i - y[i])**2
    cost = cost / (2*m)
    return cost

# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(X_train, y_train, theta_init, theta0_init)
print(f'Cost at optimal theta : {cost}')


def compute_gradient(X, y, theta, theta0):
    """Computes the gradient for linear regression

    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        theta (ndarray (n, )): model parameter
        theta0 (scalar): model parameter
    
    Returns:
        dj_dtheta (ndarray (n,)): The gradient of the cost with respect to theta
        dj_dtheta0 (scalar0)    : The gradient of the cost with respect to theta0
    """
    m,n = X.shape
    dj_dtheta = np.zeros((n,))
    dj_dtheta0 = 0.
    
    for i in range(m):
        err = (np.dot(X[i], theta) + theta0) - y[i]
        dj_dtheta += err * X[i]
        dj_dtheta0 += err
    dj_dtheta = dj_dtheta / m
    dj_dtheta0 = dj_dtheta0 / m
    
    return dj_dtheta, dj_dtheta0

#Compute and display gradient
tmp_dj_dtheta, tmp_dj_dtheta0 = compute_gradient(X_train, y_train, theta_init, theta0_init)
print(f'dj_dtheta at initial theta, theta0: {tmp_dj_dtheta}')
print(f'dj_dtheta0 at initial theta, theta0: {tmp_dj_dtheta0}')



def gradient_descent(X, y, theta_in, theta0_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha.

    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        theta_in (ndarray (n,)): initial model parameters
        theta0_in (scalar): initial model parameter
        cost_function (func): function to compute cost (J)
        gradient_function (func): function to compute the gradient
        alpha (float): Learning rate
        num_iters (int): number of iterations to run gradient descent 
        
    Returns:
        theta (ndarray (n,)) : Updated values of parameters
        theta0 (scalar)      : Updated value of parameter
    """

    #An array to store cost J and theta's at each iteration primarily for graphing later
    J_history = []
    theta = copy.deepcopy(theta_in)
    theta0 = theta_in
    
    for i in range(num_iters):
        
        #Calculate the gradient and update the parameters
        dj_dtheta, dj_dtheta0 = gradient_function(X, y, theta, theta0)
        
        #Update Parameters using theta, theta0, alpha, and gradient
        theta = theta - alpha * dj_dtheta
        theta0 = theta0 - alpha * dj_dtheta0
        
        #Save cost J at each iteration
        if i < 100000:            #prevents resource exhaustion
            J_history.append(cost_function(X, y, theta, theta0))

        #Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    
    return theta, theta0, J_history 


#initialize parameters
initial_theta = np.zeros_like(theta_init)
initial_theta0 = 0.     
#some gradient descent settings
iterations = 1000
alpha = 5.0e-7
#run gradient descent
theta_final, theta0_final, J_hist = gradient_descent(X_train, y_train, initial_theta, initial_theta0, compute_cost, compute_gradient, alpha, iterations)
print(f"theta, theta0 found by gradient descent: {theta_final:0.2f},{theta0_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], theta_final) + theta0_final:0.2f}, target value: {y_train[i]}")


# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()