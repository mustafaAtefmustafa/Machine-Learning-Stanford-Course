import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Load data
path = 'D:\\Andrew NG Tasks\\Linear Regression with one variable\\ML.csv'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# Separate Data (getx, gety)
cols = data.shape[1] # ---> number of columns
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

# Plot the data
fig, ax = plt.subplots()
ax.scatter(X, y, c='blue',label='training data')

# Add column of ones
X.insert(0, 'Ones', 1)
#print('X data = \n', X.head(10))
#print('**********************')
#print('y data = \n', y.head(10))
#print('**********************')

# Convert from data frames to numpy matrices.
x = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
#print('x = \n', x)
#print('y = \n', y)
#print('theta = \n', theta)

# Cost Function.
def cost_function(x, y, theta):
    z = np.power(((x * theta.T) - y), 2)
    m = len(x)
    return (1 / (2*m)) * np.sum(z)

print('Cost at theta [0 0] = \n', cost_function(x,y,theta))

# Gradient Descent.
def gradient_descent(x, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    m = len(x)
    for i in range(iters):
        error = (x * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, x[:,j])
            temp[0,j] = theta[0,j] - ((alpha/m)) * np.sum(term)
        
        theta = temp
        cost[i] = cost_function(x,y,theta)
    return theta, cost

# Initialize variables for learning rate and iterations.
alpha = 0.01
iters = 1000

# Perform gradient descent to fit the model.
g, cost = gradient_descent(x,y,theta,alpha,iters)
print('trained theta =\n', g)
#print('cost = \n', cost[0:50])
print('cost of trained theta = \n', cost_function(x,y,g))

# Get best fit line.
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

# Draw the line.
ax.plot(x,f,'r',label='prediction')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted profit vs population size')
plt.show()