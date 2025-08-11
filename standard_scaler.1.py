# StandardScaler example in Python
# --------------------------------
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example dataset: rows = samples, columns = features
# Let's say column 1 = height (cm), column 2 = weight (kg)
data = np.array([
    [150, 50],
    [160, 60],
    [170, 80],
    [180, 90]
])

print("Original data:\n", data)

# Create the StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the data and transform it
scaled_data = scaler.fit_transform(data)

print("\nScaled data (mean=0, std=1):\n", scaled_data)

# You can also check the mean and standard deviation after scaling
print("\nMeans after scaling:", scaled_data.mean(axis=0))
print("Standard deviations after scaling:", scaled_data.std(axis=0))
#OUTPUT
#[[150  50]
 [160  60]
 [170  80]
 [180  90]]

#Scaled data (mean=0, std=1):
 [[-1.34164079 -1.26491106]
 [-0.4472136  -0.63245553]
 [ 0.4472136   0.63245553]
 [ 1.34164079  1.26491106]]

#Means after scaling: [0.00000000e+00 5.55111512e-17]
3Standard deviations after scaling: [1. 1.]

