#!/usr/bin/env python
# coding: utf-8

# In[609]:


#Aaron Skow
#ECGR 4105-HW0
#9/17/21


# In[610]:


import numpy as np
import pandas as panda
import matplotlib.pyplot as plt


# In[611]:


datafile = panda.read_csv(r'C:\Users\Aaron\Documents\D3.csv')
datafile.head()
datalength= len(datafile)


# In[612]:


X1 = datafile.values[:,0]
X2 = datafile.values[:,1]
X3 = datafile.values[:,2]
Y = datafile.values[:,3]
outputLength = len(Y)
print(Y)


# In[613]:


plt.scatter(X1,Y,color = 'red', marker = '+')
plt.grid()
plt.rcParams["figure.figsize"] = (5,5)
plt.xlabel('X1 Input Data')
plt.ylabel('Y Output Data')
plt.title('X1 vs Y Data')


# In[614]:


plt.scatter(X2,Y,color = 'blue', marker = '+')
plt.grid()
plt.rcParams["figure.figsize"] = (5,5)
plt.xlabel('X2 Input Data')
plt.ylabel('Y Output Data')
plt.title('X2 vs Y Data')


# In[615]:


plt.scatter(X3,Y,color = 'green', marker = '+')
plt.grid()
plt.rcParams["figure.figsize"] = (5,5)
plt.xlabel('X3 Input Data')
plt.ylabel('Y Output Data')
plt.title('X3 vs Y Data')


# In[616]:


onesMatrix = np.ones((outputLength,1))
X1Mat = X1.reshape(outputLength,1)
X2Mat = X2.reshape(outputLength,1)
X3Mat = X3.reshape(outputLength,1)


# In[617]:


X1 = np.hstack((onesMatrix, X1Mat))
X2 = np.hstack((onesMatrix, X2Mat))
X3 = np.hstack((onesMatrix, X3Mat))
X4 = np.hstack((onesMatrix, X1Mat, X2Mat, X3Mat))


# In[618]:


theta = np.zeros(2)
thetatwo = np.zeros(4)
print(theta)
print(thetatwo)


# In[619]:


def cost_function(X, Y, theta):
    predictions = X.dot(theta)
    error = np.subtract(predictions, Y)
    sqrErrors = np.square(error)
    J = 1 / (2 * outputLength) * np.sum(sqrErrors)
    
    return J


# In[635]:


cost1 = cost_function(X1, Y, theta)
cost2 = cost_function(X2, Y, theta)
cost3 = cost_function(X3, Y, theta)
print('cost1 = ',cost1)
print('cost2 = ',cost2)
print('cost3 = ',cost3)


# In[636]:


cost4 = cost_function(X4, Y, thetatwo)
print('cost4 = ',cost4)


# In[622]:


def gradient_function(X, Y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        predictions = X.dot(theta)
        error = np.subtract(predictions, Y)
        sum_delta = (alpha / outputLength) * X.transpose().dot(error);
        theta = theta - sum_delta;
        cost_history[i] = cost_function(X, Y, theta)
        
    return theta, cost_history


# In[623]:


theta = [0., 0.]
thetatwo = [0., 0., 0., 0.]
iterations = 1500;
alpha = 0.05;


# In[637]:


theta, cost_history = gradient_function(X1, Y, theta, alpha, iterations)
print('theta1 = ', theta)


# In[625]:


plt.scatter(X1[:, 1], Y, color = 'red', marker = '+', label = 'Training Data')
plt.plot(X1[:, 1], X1.dot(theta), color = 'green', label = 'Linear Regression')

plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('X1 Input Data')
plt.ylabel('Y Output Data')
plt.title('Linear Regression Fit X1')
plt.legend()


# In[626]:


plt.plot(range(1, iterations + 1), cost_history, color = 'blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent X1')


# In[627]:


theta, cost_history = gradient_function(X2, Y, theta, alpha, iterations)
print('Theta2 = ',theta)


# In[628]:


plt.scatter(X2[:, 1], Y, color = 'blue', marker = '+', label = 'Training Data')
plt.plot(X2[:, 1], X2.dot(theta), color = 'green', label = 'Linear Regression')

plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('X2 Input Data')
plt.ylabel('Y Output Data')
plt.title('Linear Regression Fit X2')
plt.legend()


# In[629]:


plt.plot(range(1, iterations + 1), cost_history, color = 'blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent X2')


# In[630]:


theta, cost_history = gradient_function(X3, Y, theta, alpha, iterations)
print('theta3 = ', theta)


# In[631]:


plt.scatter(X3[:, 1], Y, color = 'black', marker = '+', label = 'Training Data')
plt.plot(X3[:, 1], X3.dot(theta), color = 'green', label = 'Linear Regression')

plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('X3 Input Data')
plt.ylabel('Y Output Data')
plt.title('Linear Regression Fit X3')
plt.legend()


# In[632]:


plt.plot(range(1, iterations + 1), cost_history, color = 'blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent X3')


# In[638]:


thetatwo, cost_history = gradient_function(X4, Y, thetatwo, alpha, iterations)
print('thetatwo = ', thetatwo)
print()


# In[634]:


plt.plot(range(1, iterations + 1), cost_history, color = 'blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent X4')


# In[643]:


#predicted values of Y for final linear model
A = [1,1,1,1]
B = [1,2,0,4]
C = [1,3,2,1]

y1 = A * thetatwo
y2 = B * thetatwo
y3 = C * thetatwo

pred1 = y1[0] + y1[1] + y1[2] + y1[3]
pred2 = y2[0] + y2[1] + y2[2] + y2[3]
pred3 = y3[0] + y3[1] + y3[2] + y3[3]

print(y1)
print(y2)
print(y3)

print(pred1)
print(pred2)
print(pred3)

