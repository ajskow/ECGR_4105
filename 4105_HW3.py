#!/usr/bin/env python
# coding: utf-8

# In[33]:


#Aaron Skow
#ECGR - 4105 - HW3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[34]:


#Load dataset
breast = load_breast_cancer()


# In[35]:


#Format dataset
breast_data = breast.data
breast_data.shape


# In[36]:


#Create dataframe from dataset
breast_input = pd.DataFrame(breast_data)
breast_input.head()


# In[37]:


#Create labels for dataset
breast_labels = breast.target
breast_labels.shape


# In[38]:


labels = np.reshape(breast_labels,(569,1))


# In[39]:


#Finalize format of dataset
final_breast_data = np.concatenate([breast_data,labels],axis = 1)
final_breast_data.shape


# In[40]:


breast_dataset = pd.DataFrame(final_breast_data)


# In[41]:


#Define dataset features
features = breast.feature_names
features


# In[42]:


feature_labels = np.append(features,'label')


# In[43]:


breast_dataset.columns = feature_labels


# In[44]:


breast_dataset.head()


# In[45]:


#Assign binary values to relevant labels
breast_dataset['label'].replace(0, 'Benign', inplace = True)
breast_dataset['label'].replace(1, 'Malignant', inplace = True)


# In[46]:


breast_dataset.tail()


# In[47]:


#Problem 1
#Separate data into input and output
X = breast_dataset.iloc[:, :30].values
Y = breast_dataset.iloc[:, 30].values


# In[48]:


#Split into training/test data
rand = np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = rand)


# In[49]:


#Standardize training/test data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[50]:


#Create logistic regression object
classifier = LogisticRegression(random_state = 0, max_iter = 3000)
classifier.fit(X_train, Y_train)


# In[51]:


#Run logistic regression
LogisticRegression(random_state = 0, max_iter = 3000)


# In[52]:


#Get predicted values from model
Y_pred = classifier.predict(X_test)


# In[53]:


#Display logistic regression metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))


# In[ ]:





# In[54]:


#Problem 2
#initialize PCA and fit to data
pca = PCA()
principalComp = pca.fit_transform(X)


# In[55]:


#define K and goal variables
K = principalComp.shape[1]
BestK = 0
BestAcc = 0


# In[56]:


#initialize metrics
accuracy = np.zeros(K)
precision = np.zeros(K)
recall = np.zeros(K)


# In[57]:


#Split into training/test data
rand = np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(principalComp, Y, test_size = 0.2, train_size = 0.8, random_state = rand)


# In[58]:


#Train the model for each component and update metrics
for i in range(K):
    classifier.fit(X_train[:, :i+1], Y_train)
    Y_pred = classifier.predict(X_test[:, :i+1])
    
    accuracy[i] = metrics.accuracy_score(Y_test, Y_pred)
    precision[i] = metrics.precision_score(Y_test, Y_pred, pos_label = 'Malignant')
    recall[i] = metrics.recall_score(Y_test, Y_pred, pos_label = 'Malignant')
    
    if accuracy[i] > BestAcc:
        BestAcc = accuracy[i]
        BestK = i+1


# In[59]:


#Plot metric values over all values of K
plt.figure()
plt.plot(np.linspace(1, K, K), accuracy, color='blue', label='Accuracy')
plt.plot(np.linspace(1, K, K), precision, color='red', label='Precision')
plt.plot(np.linspace(1, K, K), recall, color='green', label='Recall')
plt.rcParams['figure.figsize'] = (10, 8)
plt.grid()

plt.title('Metric scores over 30 K values')
plt.ylabel('Values of Metric scores')
plt.xlabel('Values of K')
plt.legend()


# In[60]:


#Display metric values and optimal K
print('Best K:', BestK)
print('Accuracy:', BestAcc)
print('Precision:', precision[BestK - 1])
print('Recall:', recall[BestK - 1])


# In[ ]:





# In[61]:


#Problem 3
#Create LDA object and fit to training data
LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train, Y_train)


# In[62]:


#Obtain predicted output using LDA
Y_pred = LDA.predict(X_test)


# In[63]:


#Display metrics including 'Malignant' and 'Benign' for positive label
print('Accuracy:', metrics.accuracy_score(Y_test, Y_pred))
print('Precision for "Malignant":', metrics.precision_score(Y_test, Y_pred, pos_label = 'Malignant'))
print('recall for "Malignant":', metrics.recall_score(Y_test, Y_pred, pos_label = 'Malignant'))
print('Precision for "Benign":', metrics.precision_score(Y_test, Y_pred, pos_label = 'Benign'))
print('recall for "Benign":', metrics.recall_score(Y_test, Y_pred, pos_label = 'Benign'))


# In[ ]:





# In[64]:


#Problem 4
#Create LDA object and fit to data
LDA2 = LDA.fit_transform(X, Y)


# In[65]:


#Split into training/test data
rand = np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(LDA2, Y, test_size = 0.2, train_size = 0.8, random_state = rand)


# In[66]:


#Fit logistic regression model to training data
classifier.fit(X_train, Y_train)


# In[67]:


#Get predicted output of model
Y_pred = classifier.predict(X_test)


# In[68]:


#Display metrics including 'Malignant' and 'Benign' for positive label
print('Accuracy:', metrics.accuracy_score(Y_test, Y_pred))
print('Precision for "Malignant":', metrics.precision_score(Y_test, Y_pred, pos_label = 'Malignant'))
print('recall for "Malignant":', metrics.recall_score(Y_test, Y_pred, pos_label = 'Malignant'))
print('Precision for "Benign":', metrics.precision_score(Y_test, Y_pred, pos_label = 'Benign'))
print('recall for "Benign":', metrics.recall_score(Y_test, Y_pred, pos_label = 'Benign'))


# In[ ]:





# In[ ]:




