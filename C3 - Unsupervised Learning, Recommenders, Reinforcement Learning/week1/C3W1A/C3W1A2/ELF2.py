from scipy.stats import skewnorm
import numpy as np
import matplotlib.pyplot as plt

numValues = 10000
maxValue = 100
skewness = 20

random = skewnorm.rvs(a=skewness, loc=maxValue, size=numValues)

random = random - min(random) #Shift the set so that the value is equal to zero
random = random / max(random) #Standardize all the values between 0 and 1
random = random * maxValue    #Multipy the standardized values by the maximum value

x = random
plt.hist(x, bins=50, color= 'r')


plt.hist(x**0.5, bins = 50, color= 'r')

y = np.log(x + 7)
plt.hist(y, bins=50) 



