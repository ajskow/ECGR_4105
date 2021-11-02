#!/usr/bin/env python
# coding: utf-8

# In[66]:


#Aaron Skow
#ECGR - 4105 - HW2
#ID: 801161395
#10/28/21

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


# In[41]:


#Get data from diabetes csv file
dataset = pd.read_csv(r'C:\Users\Aaron\Downloads\Diabetes.csv')


# In[42]:


dataset.head()


# In[43]:


#Separate data into input and output
X = dataset.iloc[:, [0,1,2,3,4,5,6,7]].values
Y = dataset.iloc[:, 8].values


# In[44]:


#Problem 1
#Split into training/test data
rand = np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = rand)


# In[45]:


#Standardize training/test data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[46]:


#Create logistic regression object
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)


# In[47]:


LogisticRegression(random_state = 0)


# In[48]:


#Get predicted values from model
Y_pred = classifier.predict(X_test)


# In[49]:


#Create confusion matrix for results
cnf_matrix = confusion_matrix(Y_test, Y_pred)
cnf_matrix


# In[50]:


#Display logistic regression metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred)) 
print("Precision:",metrics.precision_score(Y_test, Y_pred)) 
print("Recall:",metrics.recall_score(Y_test, Y_pred)) 


# In[51]:


#Plot results of Linear regression model
class_names=[0,1]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[52]:


#Problem 2
#Create Naive Bayes object
classifier2 = GaussianNB()
classifier2.fit(X_train, Y_train)


# In[53]:


#Get predicted values from model
Y2_pred = classifier2.predict(X_test)


# In[54]:


#Create confusion matrix and metrics
cm = confusion_matrix(Y_test, Y2_pred)
ac = accuracy_score(Y_test, Y2_pred)


# In[55]:


#Display Naive Bayes metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y2_pred)) 
print("Precision:",metrics.precision_score(Y_test, Y2_pred)) 
print("Recall:",metrics.recall_score(Y_test, Y2_pred)) 


# In[56]:


#Display confusion matrix values
cm


# In[57]:


#Plot results of Naive Bayes model
class_names=[0,1]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names)

#Create heatmap 
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[58]:


#Problem 3
#Kfold with k = 5
#Implement K-fold cross validation for Linear regression
metrics = ['accuracy', 'precision', 'recall']
kfold = KFold(shuffle = True, random_state = 0, n_splits = 5)
results = cross_validate(classifier, X, Y, cv = kfold, scoring = metrics, n_jobs = -1)


# In[59]:


#Display Linear regression metrics for K = 5
print("Accuracy:", results['test_accuracy']) 
print("Precision:", results['test_precision'])
print("Recall:", results['test_recall'])


# In[60]:


#kfold with k = 10
#Implement K-fold cross validation for Linear regression
kfold2 = KFold(shuffle = True, random_state = 0, n_splits = 10)
results2 = cross_validate(classifier, X, Y, cv = kfold2, scoring = metrics, n_jobs = -1)


# In[61]:


#Display Linear regression metrics for K = 10
print("Accuracy:", results2['test_accuracy']) 
print("Precision:", results2['test_precision'])
print("Recall:", results2['test_recall'])


# In[62]:


#Problem 4
#Kfold with k = 5
#Implement K-fold cross validation for Naive Bayes
results3 = cross_validate(classifier2, X, Y, cv = kfold, scoring = metrics, n_jobs = -1)


# In[63]:


#Display Naive Bayes metrics for K = 5
print("Accuracy:", results3['test_accuracy']) 
print("Precision:", results3['test_precision'])
print("Recall:", results3['test_recall'])


# In[64]:


#Kfold with k = 10
#Implement K-fold cross validation for Naive Bayes
results4 = cross_validate(classifier2, X, Y, cv = kfold2, scoring = metrics, n_jobs = -1)


# In[65]:


#Display Naive Bayes metrics for K = 10
print("Accuracy:", results4['test_accuracy']) 
print("Precision:", results4['test_precision'])
print("Recall:", results4['test_recall'])

