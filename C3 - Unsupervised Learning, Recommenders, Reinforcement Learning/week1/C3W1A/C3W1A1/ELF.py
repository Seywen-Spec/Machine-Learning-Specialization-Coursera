import numpy as np
import matplotlib.pyplot as plt
from utils import *


def find_closest_centroids(X, centroids):
    """Computes the centroid membreships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): k centroids
        
    Returns:
        idx (array_like): (m,) closest centroids
    """
    
    #Set K
    K = centroids.shape[0]
    
    idx = np.zeros(X.shape[0], dtype=int)
    
    for i in range(X.shape[0]):
        distance = []
        for j in range(K):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
        
        idx[i] = np.argmin(distance)
    
    return idx

X = ('load data')

print("First five elements of X are:\n", X[:5]) 
print('The shape of X is:', X.shape) 

    
    
# Select an initial set of centroids (3 Centroids)
initial_centroids = np.array([[3,3], [6,2], [8,5]])

# Find closest centroids using initial_centroids
idx = find_closest_centroids(X, initial_centroids)

# Print closest centroids for the first three elements
print("First three elements in idx are:", idx[:3])



def compute_centroids(X, idx, K):
    """Returnds the new centroids by computing the means of the data points assigned to each centroid.

    Args:
        X (ndarray): (m,n) Data Points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i 
        K (int): number of centroids
    
    Returns: 
        centroids (ndarray): (K, n) New centroids computed
    """
    
    m, n = X.shape
    
    centroids = np.zeros((K, n))
    
    for k in range (K):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis=0)
        
    return centroids

K = 3
centroids = compute_centroids(X, idx, K)

print("The centroids are:", centroids)



# You do not need to implement anything for this part

def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    
    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        # Optionally plot progress
        if plot_progress:
            #plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx

# Load an example dataset
#X = load_data()

# Set initial centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]])
K = 3

# Number of iterations
max_iters = 10

centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)




def kMeans_init_centroids(X, K):
    """This function initializes K centroids that are to be used in K-Means on teh dataset X

    Args:
        X (ndarray): Data points
        K (int): number of centroids/clusters
    Returns:
        centroids (ndarray) : Initialzed centroids
    """

    # Randomly reorder the indices of examples 
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as centroids
    centroids = X[randidx[:K]]

    return centroids


# Load an image of a bird
original_img = plt.imread('bird_small.png')

# Visualizing the image
plt.imshow(original_img)

print("Shape of original_img is:", original_img.shape)

# Divide by 255 so that all values are in the range 0 - 1
original_img = original_img / 255

# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))


# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16                       
max_iters = 10               

# Using the function you have implemented above. 
initial_centroids = kMeans_init_centroids(X_img, K) 

# Run K-Means - this takes a couple of minutes
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters) 

print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])


print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])

# Represent image in terms of indices
X_recovered = centroids[idx, :] 

# Reshape recovered image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape) 


# Display original image
fig, ax = plt.subplots(1,2, figsize=(8,8))
plt.axis('off')

ax[0].imshow(original_img*255)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_recovered*255)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()


