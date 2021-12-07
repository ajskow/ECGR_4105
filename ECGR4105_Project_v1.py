#!/usr/bin/env python
# coding: utf-8

# In[172]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


# In[173]:


#Load dataset
Grid_QOL = pd.read_csv(r'C:\Users\Aaron\Downloads\combinedDataThree.csv')


# In[174]:


Grid_QOL.head()


# In[237]:


#Separate data into input and output
X = Grid_QOL.iloc[:,[0,2,3,4,5,6]].values
Y = Grid_QOL.iloc[:, 1].values


# In[238]:


#Split into training and test data
rand = np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = rand)


# In[239]:


#Standardize training/test data
sc_X = StandardScaler()
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[240]:


#Create Logistic Regression object
classifier = LogisticRegression(max_iter=15000, random_state=0)
classifier.fit(X_train,Y_train)


# In[241]:


LogisticRegression(random_state = 0)


# In[242]:


#Get predited values from model
Y_pred = classifier.predict(X_test)


# In[243]:


#Create confusion matrix for results
cnf_matrix = confusion_matrix(Y_test, Y_pred)


# In[244]:


#Display logistic regression metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred)) 
print("Precision:",metrics.precision_score(Y_test, Y_pred,average = 'weighted')) 
print("Recall:",metrics.recall_score(Y_test, Y_pred,average = 'weighted')) 


# In[245]:


#Plot results of Linear regression model
class_names=[0,1,2,3,4,5]
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


# In[ ]:





# In[ ]:





# In[246]:


#Separate data into input and output
X = Grid_QOL.iloc[:,[0,1,3,4,5,6]].values
Y = Grid_QOL.iloc[:, 2].values


# In[247]:


#Split into training and test data
rand = np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = rand)


# In[248]:


#Standardize training/test data
sc_X = StandardScaler()
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[249]:


#Create Logistic Regression object
classifier = LogisticRegression(max_iter=5000, random_state=0)
classifier.fit(X_train,Y_train)


# In[250]:


#Get predited values from model
Y_pred2 = classifier.predict(X_test)


# In[251]:


#Create confusion matrix for results
cnf_matrix = confusion_matrix(Y_test, Y_pred2)


# In[252]:


#Display logistic regression metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred2)) 
print("Precision:",metrics.precision_score(Y_test, Y_pred2,average = 'weighted')) 
print("Recall:",metrics.recall_score(Y_test, Y_pred2,average = 'weighted')) 


# In[253]:


#Plot results of Linear regression model
class_names=[0,1,2,3,4,5]
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


# In[ ]:





# In[229]:


#Separate data into input and output
X = Grid_QOL.iloc[:,[0,1,2,4,5,6]].values
Y = Grid_QOL.iloc[:, 3].values


# In[230]:


#Split into training and test data
rand = np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = rand)


# In[231]:


#Standardize training/test data
sc_X = StandardScaler()
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[232]:


#Create Logistic Regression object
classifier = LogisticRegression(max_iter=5000, random_state=0)
classifier.fit(X_train,Y_train)


# In[233]:


#Get predited values from model
Y_pred3 = classifier.predict(X_test)


# In[234]:


#Create confusion matrix for results
cnf_matrix = confusion_matrix(Y_test, Y_pred3)


# In[235]:


#Display logistic regression metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred3)) 
print("Precision:",metrics.precision_score(Y_test, Y_pred3,average = 'weighted')) 
print("Recall:",metrics.recall_score(Y_test, Y_pred3,average = 'weighted')) 


# In[236]:


#Plot results of Linear regression model
class_names=[0,1,2,3,4,5]
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


# In[ ]:





# In[254]:


#Separate data into input and output
X = Grid_QOL.iloc[:,[0,1,2,3,5,6]].values
Y = Grid_QOL.iloc[:, 4].values


# In[255]:


#Split into training and test data
rand = np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = rand)


# In[256]:


#Standardize training/test data
sc_X = StandardScaler()
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[257]:


#Create Logistic Regression object
classifier = LogisticRegression(max_iter=5000, random_state=0)
classifier.fit(X_train,Y_train)


# In[258]:


#Get predited values from model
Y_pred4 = classifier.predict(X_test)


# In[259]:


#Create confusion matrix for results
cnf_matrix = confusion_matrix(Y_test, Y_pred4)


# In[276]:


#Display logistic regression metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred4)) 
print("Precision:",metrics.precision_score(Y_test, Y_pred4,average='weighted')) 
print("Recall:",metrics.recall_score(Y_test, Y_pred4,average = 'weighted')) 


# In[261]:


#Plot results of Linear regression model
class_names=[0,1,2,3,4,5]
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


# In[ ]:





# In[270]:


#Create Naive Bayes object
classifier2 = GaussianNB()
classifier2.fit(X_train, Y_train)


# In[271]:


#Get predicted values from model
Y2_pred = classifier2.predict(X_test)


# In[272]:


#Create confusion matrix and metrics
cm = confusion_matrix(Y_test, Y2_pred)
ac = accuracy_score(Y_test, Y2_pred)


# In[273]:


#Display Naive Bayes metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y2_pred)) 
#print("Precision:",metrics.precision_score(Y_test, Y2_pred)) 
#print("Recall:",metrics.recall_score(Y_test, Y2_pred)) 


# In[274]:


#Plot results of Naive Bayes model
class_names=[0,1,2,3,4,5,6,7]
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


# In[ ]:





# In[ ]:




