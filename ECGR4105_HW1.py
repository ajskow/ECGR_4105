#!/usr/bin/env python
# coding: utf-8

# In[710]:


import numpy as np
import pandas as panda
import matplotlib.pyplot as plt
import seaborn as sns


# In[711]:


#Import data from csv
housing = panda.DataFrame(panda.read_csv(r'C:\Users\Aaron\Downloads\Housing.csv'))
housing.head()
datalength = len(housing)


# In[712]:


def cost_function(X, Y, theta):
    predictions = X.dot(theta)
    error = np.subtract(predictions, Y)
    sqrErrors = np.square(error)
    J = 1 / (2 * datalength) * np.sum(sqrErrors)
    
    return J


# In[713]:


def cost_function2(X, Y, theta, lamb):
    predictions = X.dot(theta)
    error = np.subtract(predictions, Y)
    sqrErrors = np.square(error)
    J = 1 / (2 * datalength) * (np.sum(sqrErrors) + lamb * np.sum(np.square(theta)))
    
    return J


# In[714]:


def gradient_function(X, Y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        predictions = X.dot(theta)
        error = np.subtract(predictions, Y)
        sum_delta = (alpha / datalength) * X.transpose().dot(error)
        theta = theta - sum_delta
        cost_history[i] = cost_function(X, Y, theta)
        
    return theta, cost_history


# In[715]:


def gradient_function2(X, Y, theta, alpha, iterations, lamb):
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        predictions = X.dot(theta)
        error = np.subtract(predictions, Y)
        sum_delta = (alpha / datalength) * (X.transpose().dot(error) + lamb * theta)
        theta = theta - sum_delta
        cost_history[i] = cost_function2(X, Y, theta, lamb)
        
    return theta, cost_history


# In[716]:


housing.shape
housing.info()


# In[717]:


housing.describe()


# In[718]:


#Assign string values to binary values
varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

def binary_map(x):
    return x.map({'yes' : 1, 'no' : 0})

housing[varlist] = housing[varlist].apply(binary_map)

housing.head()


# In[719]:


#Question 1A---------------------


# In[720]:


#Divide datafile into training and test sets
from sklearn.model_selection import train_test_split

rand = np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = rand)
df_train.shape


# In[721]:


df_test.shape


# In[722]:


#specify data for training and test sets
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
df_Newtrain = df_train[num_vars]
df_Newtest = df_test[num_vars]
df_Newtrain.head()


# In[723]:


df_Newtrain.shape


# In[724]:


#NON-Scaled training input and output
Y_df_Newtrain = df_Newtrain.pop('price')
X_df_Newtrain = df_Newtrain

#NON-Scaled test input and output
Y_df_Newtest = df_Newtest.pop('price')
X_df_Newtest = df_Newtest


# In[725]:


Nonscaled_train_theta = [0., 0., 0., 0., 0.]
Nonscaled_test_theta = [0., 0., 0., 0., 0.]
Nonscaled_iterations = 1500;
Nonscaled_alpha = 0.05


# In[726]:


Nonscaled_train_cost = cost_function(X_df_Newtrain, Y_df_Newtrain, Nonscaled_train_theta)
Nonscaled_test_cost = cost_function(X_df_Newtest, Y_df_Newtest, Nonscaled_test_theta)
print(Nonscaled_train_cost, Nonscaled_test_cost)


# In[727]:


Nonscaled_train_theta, Nonscaled_train_cost_history = gradient_function(X_df_Newtrain, Y_df_Newtrain, Nonscaled_train_theta, Nonscaled_alpha, Nonscaled_iterations)
print('Nonscaled_train_theta = ', Nonscaled_train_theta)


# In[728]:


Nonscaled_test_theta, Nonscaled_test_cost_history = gradient_function(X_df_Newtest, Y_df_Newtest, Nonscaled_test_theta, Nonscaled_alpha, Nonscaled_iterations)
print('Nonscaled_test_theta = ', Nonscaled_test_theta)


# In[729]:


plt.plot(range(1, Nonscaled_iterations + 1), Nonscaled_train_cost_history, color = 'blue')
plt.plot(range(1, Nonscaled_iterations + 1), Nonscaled_test_cost_history, color = 'red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Non-Scaled Convergence of gradient descent')


# In[730]:


#Question 1B---------------------


# In[731]:


#specify data for training and test sets with more parameters
num_vars2 = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'price']
df_Newtrain2 = df_train[num_vars2]
df_Newtest2 = df_test[num_vars2]


# In[732]:


#NON-Scaled training input and output
Y_df_Newtrain2 = df_Newtrain2.pop('price')
X_df_Newtrain2 = df_Newtrain2

#NON-Scaled test input and output
Y_df_Newtest2 = df_Newtest2.pop('price')
X_df_Newtest2 = df_Newtest2


# In[733]:


Nonscaled_train_theta2 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
Nonscaled_test_theta2 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
Nonscaled_iterations2 = 1500;
Nonscaled_alpha2 = 0.05


# In[734]:


Nonscaled_train_cost2 = cost_function(X_df_Newtrain2, Y_df_Newtrain2, Nonscaled_train_theta2)
Nonscaled_test_cost2 = cost_function(X_df_Newtest2, Y_df_Newtest2, Nonscaled_test_theta2)
print(Nonscaled_train_cost2, Nonscaled_test_cost2)


# In[735]:


Nonscaled_train_theta2, Nonscaled_train_cost_history2 = gradient_function(X_df_Newtrain2, Y_df_Newtrain2, Nonscaled_train_theta2, Nonscaled_alpha2, Nonscaled_iterations2)
print('Nonscaled_train_theta2 = ', Nonscaled_train_theta2)


# In[736]:


Nonscaled_test_theta2, Nonscaled_test_cost_history2 = gradient_function(X_df_Newtest2, Y_df_Newtest2, Nonscaled_test_theta2, Nonscaled_alpha2, Nonscaled_iterations2)
print('Nonscaled_test_theta2 = ', Nonscaled_test_theta2)


# In[737]:


plt.plot(range(1, Nonscaled_iterations2 + 1), Nonscaled_train_cost_history2, color = 'blue')
plt.plot(range(1, Nonscaled_iterations2 + 1), Nonscaled_test_cost_history2, color = 'red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Non-Scaled Convergence of gradient descent with extra parameters')


# In[738]:


#Question 2A Normalization---------------------


# In[739]:


#Divide datafile into training and test sets
from sklearn.model_selection import train_test_split

rand = np.random.seed(0)
scaled_df_train, scaled_df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = rand)
scaled_df_train.shape


# In[740]:


scaled_df_test.shape


# In[741]:


#specify data for training and test sets
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
Scaled_df_Newtrain = scaled_df_train[num_vars]
Scaled_df_Newtest = scaled_df_test[num_vars]
Scaled_df_Newtrain.head()


# In[742]:


Scaled_df_Newtrain.shape


# In[743]:


#Preprocess scaling of training/test datasets
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
Scaled_df_Newtrain[num_vars] = scaler.fit_transform(Scaled_df_Newtrain[num_vars])
Scaled_df_Newtest[num_vars] = scaler.fit_transform(Scaled_df_Newtest[num_vars])

Scaled_df_Newtrain.head(20)


# In[744]:


#Scaled training input and output
Y_Newtrain = Scaled_df_Newtrain.pop('price')
X_Newtrain = Scaled_df_Newtrain

#Scaled test input and output
Y_Newtest = Scaled_df_Newtest.pop('price')
X_Newtest = Scaled_df_Newtest

X_Newtrain.head()


# In[745]:


Y_Newtrain.head()


# In[746]:


ScaleLength = len(X_Newtrain)


# In[747]:


Scaled_train_theta = [0., 0., 0., 0., 0.]
Scaled_test_theta = [0., 0., 0., 0., 0.]
Scaled_iterations = 1500;
Scaled_alpha = 0.05


# In[748]:


Scaled_train_cost = cost_function(X_Newtrain, Y_Newtrain, Scaled_train_theta)
Scaled_test_cost = cost_function(X_Newtest, Y_Newtest, Scaled_test_theta)
print(Scaled_train_cost, Scaled_test_cost)


# In[749]:


Scaled_train_theta, Scaled_train_cost_history = gradient_function(X_Newtrain, Y_Newtrain, Scaled_train_theta, Scaled_alpha, Scaled_iterations)
print('Scaled_train_theta = ', Scaled_train_theta)


# In[750]:


Scaled_test_theta, Scaled_test_cost_history = gradient_function(X_Newtest, Y_Newtest, Scaled_test_theta, Scaled_alpha, Scaled_iterations)
print('Scaled_test_theta = ', Scaled_test_theta)


# In[751]:


plt.plot(range(1, Scaled_iterations + 1), Scaled_train_cost_history, color = 'blue')
plt.plot(range(1, Scaled_iterations + 1), Scaled_test_cost_history, color = 'red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Normalized Convergence of gradient descent')


# In[752]:


#Question 2A Standardization---------------------


# In[753]:


#Divide datafile into training and test sets
from sklearn.model_selection import train_test_split

rand = np.random.seed(0)
scaled_df_train2, scaled_df_test2 = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = rand)


# In[754]:


#specify data for training and test sets
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
Scaled_df_Newtrain2 = scaled_df_train2[num_vars]
Scaled_df_Newtest2 = scaled_df_test2[num_vars]


# In[755]:


#Preprocess scaling of training/test datasets
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
Scaled_df_Newtrain2[num_vars] = scaler.fit_transform(Scaled_df_Newtrain2[num_vars])
Scaled_df_Newtest2[num_vars] = scaler.fit_transform(Scaled_df_Newtest2[num_vars])


# In[756]:


#Scaled training input and output
Y_Newtrain2 = Scaled_df_Newtrain2.pop('price')
X_Newtrain2 = Scaled_df_Newtrain2

#Scaled test input and output
Y_Newtest2 = Scaled_df_Newtest2.pop('price')
X_Newtest2 = Scaled_df_Newtest2


# In[757]:


Scaled_train_theta2 = [0., 0., 0., 0., 0.]
Scaled_test_theta2 = [0., 0., 0., 0., 0.]
Scaled_iterations2 = 1500;
Scaled_alpha2 = 0.05


# In[758]:


Scaled_train_cost2 = cost_function(X_Newtrain2, Y_Newtrain2, Scaled_train_theta2)
Scaled_test_cost2 = cost_function(X_Newtest2, Y_Newtest2, Scaled_test_theta2)
print(Scaled_train_cost2, Scaled_test_cost2)


# In[759]:


Scaled_train_theta2, Scaled_train_cost_history2 = gradient_function(X_Newtrain2, Y_Newtrain2, Scaled_train_theta2, Scaled_alpha2, Scaled_iterations2)
print('Scaled_train_theta = ', Scaled_train_theta)


# In[760]:


Scaled_test_theta2, Scaled_test_cost_history2 = gradient_function(X_Newtest2, Y_Newtest2, Scaled_test_theta2, Scaled_alpha2, Scaled_iterations2)
print('Scaled_test_theta2 = ', Scaled_test_theta2)


# In[761]:


plt.plot(range(1, Scaled_iterations2 + 1), Scaled_train_cost_history2, color = 'blue')
plt.plot(range(1, Scaled_iterations2 + 1), Scaled_test_cost_history2, color = 'red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Standardized Convergence of gradient descent')


# In[762]:


#Question 2B Normalization---------------------


# In[763]:


#Divide datafile into training and test sets
from sklearn.model_selection import train_test_split

rand = np.random.seed(0)
normal_df_train, normal_df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = rand)


# In[764]:


#specify data for training and test sets with more parameters
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'price']
normal_Newtrain = normal_df_train[num_vars]
normal_Newtest = normal_df_test[num_vars]


# In[765]:


#Preprocess scaling of training/test datasets
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
normal_Newtrain[num_vars] = scaler.fit_transform(normal_Newtrain[num_vars])
normal_Newtest[num_vars] = scaler.fit_transform(normal_Newtest[num_vars])


# In[766]:


#Scaled training input and output
normal_Y_Newtrain = normal_Newtrain.pop('price')
normal_X_Newtrain = normal_Newtrain

#Scaled test input and output
normal_Y_Newtest = normal_Newtest.pop('price')
normal_X_Newtest = normal_Newtest


# In[767]:


normal_train_theta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
normal_test_theta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
normal_iterations = 1500;
normal_alpha = 0.05


# In[768]:


normal_train_cost = cost_function(normal_X_Newtrain, normal_Y_Newtrain, normal_train_theta)
normal_test_cost = cost_function(normal_X_Newtest, normal_Y_Newtest, normal_test_theta)
print(normal_train_cost, normal_test_cost)


# In[769]:


normal_train_theta, normal_train_cost_history = gradient_function(normal_X_Newtrain, normal_Y_Newtrain, normal_train_theta, normal_alpha, normal_iterations)
print('normal_train_theta = ', normal_train_theta)


# In[770]:


normal_test_theta, normal_test_cost_history = gradient_function(normal_X_Newtest, normal_Y_Newtest, normal_test_theta, normal_alpha, normal_iterations)
print('normal_test_theta = ', normal_test_theta)


# In[771]:


plt.plot(range(1, normal_iterations + 1), normal_train_cost_history, color = 'blue')
plt.plot(range(1, normal_iterations + 1), normal_test_cost_history, color = 'red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Normalized Convergence of gradient descent')


# In[772]:


#Question 2B Standardization---------------------


# In[773]:


#Divide datafile into training and test sets
from sklearn.model_selection import train_test_split

rand = np.random.seed(0)
standard_df_train, standard_df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = rand)


# In[774]:


#specify data for training and test sets with more parameters
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'price']
standard_Newtrain = standard_df_train[num_vars]
standard_Newtest = standard_df_test[num_vars]


# In[775]:


#Preprocess scaling of training/test datasets
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
standard_Newtrain[num_vars] = scaler.fit_transform(standard_Newtrain[num_vars])
standard_Newtest[num_vars] = scaler.fit_transform(standard_Newtest[num_vars])


# In[776]:


#Scaled training input and output
standard_Y_Newtrain = standard_Newtrain.pop('price')
standard_X_Newtrain = standard_Newtrain

#Scaled test input and output
standard_Y_Newtest = standard_Newtest.pop('price')
standard_X_Newtest = standard_Newtest


# In[777]:


standard_train_theta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
standard_test_theta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
standard_iterations = 1500;
standard_alpha = 0.05


# In[778]:


standard_train_cost = cost_function(standard_X_Newtrain, standard_Y_Newtrain, standard_train_theta)
standard_test_cost = cost_function(standard_X_Newtest, standard_Y_Newtest, standard_test_theta)
print(standard_train_cost, standard_test_cost)


# In[779]:


standard_train_theta, standard_train_cost_history = gradient_function(standard_X_Newtrain, standard_Y_Newtrain, standard_train_theta, standard_alpha, standard_iterations)
print('standard_train_theta = ', standard_train_theta)


# In[780]:


standard_test_theta, standard_test_cost_history = gradient_function(standard_X_Newtest, standard_Y_Newtest, standard_test_theta, standard_alpha, standard_iterations)
print('standard_test_theta = ', standard_test_theta)


# In[781]:


plt.plot(range(1, standard_iterations + 1), standard_train_cost_history, color = 'blue')
plt.plot(range(1, standard_iterations + 1), standard_test_cost_history, color = 'red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Standardized Convergence of gradient descent')


# In[782]:


#Question 3A Normalization ---------------------


# In[783]:


#Divide datafile into training and test sets
from sklearn.model_selection import train_test_split

rand = np.random.seed(0)
normalreg_df_train, normalreg_df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = rand)


# In[784]:


#specify data for training and test sets
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
normalreg_df_Newtrain = normalreg_df_train[num_vars]
normalreg_df_Newtest = normalreg_df_test[num_vars]


# In[785]:


#Preprocess scaling of training/test datasets
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
normalreg_df_Newtrain[num_vars] = scaler.fit_transform(normalreg_df_Newtrain[num_vars])
normalreg_df_Newtest[num_vars] = scaler.fit_transform(normalreg_df_Newtest[num_vars])


# In[786]:


normlen = len(normalreg_df_Newtrain)


# In[787]:


#Scaled training input and output
normalreg_Y_Newtrain = normalreg_df_Newtrain.pop('price')
normalreg_X_Newtrain = normalreg_df_Newtrain

#Scaled test input and output
normalreg_Y_Newtest = normalreg_df_Newtest.pop('price')
normalreg_X_Newtest = normalreg_df_Newtest


# In[788]:


normalreg_train_theta = [0., 0., 0., 0., 0.]
normalreg_test_theta = [0., 0., 0., 0., 0.]
normalreg_iterations = 1500;
normalreg_alpha = 0.05
normalreg_lambda = 1


# In[789]:


normalreg_train_cost = cost_function2(normalreg_X_Newtrain, normalreg_Y_Newtrain, normalreg_train_theta, normalreg_lambda)
normalreg_test_cost = cost_function(normalreg_X_Newtest, normalreg_Y_Newtest, normalreg_test_theta)
print(normalreg_train_cost, normalreg_test_cost)


# In[790]:


normalreg_train_theta, normalreg_train_cost_history = gradient_function2(normalreg_X_Newtrain, normalreg_Y_Newtrain, normalreg_train_theta, normalreg_alpha, normalreg_iterations, normalreg_lambda)
print('normalreg_train_theta = ', normalreg_train_theta)


# In[791]:


normalreg_test_theta, normalreg_test_cost_history = gradient_function(normalreg_X_Newtest, normalreg_Y_Newtest, normalreg_test_theta, normalreg_alpha, normalreg_iterations)
print('normalreg_test_theta = ', normalreg_test_theta)


# In[792]:


plt.plot(range(1, normalreg_iterations + 1), normalreg_train_cost_history, color = 'blue')
plt.plot(range(1, normalreg_iterations + 1), normalreg_test_cost_history, color = 'red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Normalized Convergence of gradient descent with penalty')


# In[793]:


#Question 3B Normalization ---------------------


# In[794]:


#Divide datafile into training and test sets
from sklearn.model_selection import train_test_split

rand = np.random.seed(0)
normalreg_df_train2, normalreg_df_test2 = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = rand)


# In[795]:


#specify data for training and test sets with more parameters
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'price']
normalreg_df_Newtrain2 = normalreg_df_train2[num_vars]
normalreg_df_Newtest2 = normalreg_df_test2[num_vars]


# In[796]:


#Preprocess scaling of training/test datasets
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
normalreg_df_Newtrain2[num_vars] = scaler.fit_transform(normalreg_df_Newtrain2[num_vars])
normalreg_df_Newtest2[num_vars] = scaler.fit_transform(normalreg_df_Newtest2[num_vars])


# In[797]:


#Scaled training input and output
normalreg_Y_Newtrain2 = normalreg_df_Newtrain2.pop('price')
normalreg_X_Newtrain2 = normalreg_df_Newtrain2

#Scaled test input and output
normalreg_Y_Newtest2 = normalreg_df_Newtest2.pop('price')
normalreg_X_Newtest2 = normalreg_df_Newtest2


# In[798]:


normalreg_train_theta2 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
normalreg_test_theta2 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
normalreg_iterations2 = 1500;
normalreg_alpha2 = 0.05
normalreg_lambda2 = 1


# In[799]:


normalreg_train_cost2 = cost_function2(normalreg_X_Newtrain2, normalreg_Y_Newtrain2, normalreg_train_theta2, normalreg_lambda2)
normalreg_test_cost2 = cost_function(normalreg_X_Newtest2, normalreg_Y_Newtest2, normalreg_test_theta2)
print(normalreg_train_cost2, normalreg_test_cost2)


# In[800]:


normalreg_train_theta2, normalreg_train_cost_history2 = gradient_function2(normalreg_X_Newtrain2, normalreg_Y_Newtrain2, normalreg_train_theta2, normalreg_alpha2, normalreg_iterations2, normalreg_lambda2)
print('normalreg_train_theta2 = ', normalreg_train_theta2)


# In[801]:


normalreg_test_theta2, normalreg_test_cost_history2 = gradient_function(normalreg_X_Newtest2, normalreg_Y_Newtest2, normalreg_test_theta2, normalreg_alpha2, normalreg_iterations2)
print('normalreg_test_theta2 = ', normalreg_test_theta2)


# In[802]:


plt.plot(range(1, normalreg_iterations2 + 1), normalreg_train_cost_history2, color = 'blue')
plt.plot(range(1, normalreg_iterations2 + 1), normalreg_test_cost_history2, color = 'red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Normalized Convergence of gradient descent with penalty')

