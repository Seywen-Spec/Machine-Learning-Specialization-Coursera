import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras 

#Custom Training loop: Gradient descent algorithm
theta = tf.Variable(3.0)
x = 1.0
y = 1.0
alpha = 0.01

iterations = 30
for iter in range(iterations):
    with tf.GradientTape() as tape: #used to compute auto-differentiation 
        h = theta * x
        J = (h - y) ** 2
    
    #Calculating gradient with respect to alpha
    [dJdtheta] = tape.gradient(J, [theta])

    #Run a step of alpha by updating the value of theta to reduce J
    theta.assign_add(-alpha * dJdtheta) #tf,variables require special handling, hence why we can't write a normal eq
    
