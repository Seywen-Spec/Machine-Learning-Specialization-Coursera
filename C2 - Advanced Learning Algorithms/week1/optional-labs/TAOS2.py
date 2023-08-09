import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense
from keras import Sequential
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


X,Y = load_coffee_data()
print(X.shape, Y.shape)


plt_roast(X, Y)

#Nromalization
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))   
print(Xt.shape, Yt.shape)   


#Tensorflow model
tf.random.set_seed(1234)  #a[pplied to achieve consistent results
model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(3, activation= 'sigmoid', name = 'layer1'),
    Dense(1, activation = 'sigmoid', name = 'layer2')
]) 
# tf.keras.Input(shape=(2,)), specifies the expected shape of the input. This allows Tensorflow to size the weights and bias parameters at this point.  This is useful when exploring Tensorflow models. This statement can be omitted in practice and Tensorflow will size the network parameters when the input data is specified in the `model.fit` statement.  
# Including the sigmoid activation in the final layer is not considered best practice. It would instead be accounted for in the loss which improves numerical stability. This will be described in more detail in a later lab.

print(model.summary())

L1_num_params = 2*3 + 3
L2_num_params = 3*1 + 1
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )


theta1, theta01 = model.get_layer('layer1').get_weights()
theta2, theta02 = model.get_layer('layer2').get_weights()
print(f'theta1{theta1.shape}:\n', theta1, f'theta01{theta01.shape}:', theta01)
print(f'theta2{theta2.shape}:\n', theta2, f'theta02{theta02.shape}:', theta02)


model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    Xt,Yt,            
    epochs=10,
)

#Updated Weights
theta1, theta01 = model.get_layer('layer1').get_weights()
theta2, theta02 = model.get_layer('layer2').get_weights()
print("theta1:\n", theta1, "theta01:\n", theta01)
print("theta2:\n", theta2, "theta02:\n", theta02)


theta1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
theta01 = np.array([-9.87, -9.28,  1.01])
theta2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
theta02 = np.array([15.54])
model.get_layer("layer1").set_weights([theta1,theta01])
model.get_layer("layer2").set_weights([theta2,theta02])


X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)


yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")

yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")