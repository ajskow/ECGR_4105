#!/usr/bin/env python
# coding: utf-8

# In[446]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings('ignore')


# In[447]:


#Load dataset
Grid_QOL = pd.read_csv(r'C:\Users\Aaron\Downloads\combinedDataTwo.csv')


# In[448]:


Grid_QOL.head()


# In[449]:


#Logistic Regression, Naive Bayes, SVM models for prediction of 'Access to Amenities Score'
#Separate data into input and output
X = Grid_QOL.iloc[:,[0,2,3,4,5]].values
Y = Grid_QOL.iloc[:, 1].values


# In[450]:


#Split into training and test data
rand = np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = rand)


# In[451]:


#Standardize training/test data
sc_X = StandardScaler()
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[452]:


#Create Logistic Regression object
LR = LogisticRegression(max_iter=15000, random_state=0)
LR.fit(X_train,Y_train)


# In[453]:


#Create Naive Bayes object
NB = GaussianNB()
NB.fit(X_train, Y_train)


# In[454]:


#Create SVM Object
SVM = SVC()
SVM.fit(X_train,Y_train)


# In[455]:


#Get predited values from Logistic Regression model
LR_pred = LR.predict(X_test)


# In[456]:


#Get predicted values from Naive Bayes model
NB_pred = NB.predict(X_test)


# In[457]:


#Get predicted values from SVM model
SVM_pred = SVM.predict(X_test)


# In[458]:


#Create confusion matrix for results
LR_cnf_matrix = confusion_matrix(Y_test, LR_pred)


# In[459]:


#Create confusion matrix for Naive Bayes results
NB_cnf_matrix = confusion_matrix(Y_test, NB_pred)


# In[460]:


#Create confusion matrix for SVM results
SVM_cnf_matrix = confusion_matrix(Y_test, SVM_pred)


# In[461]:


#Display logistic Regression metrics
print("Accuracy:",metrics.accuracy_score(Y_test, LR_pred)) 
print("Precision:",metrics.precision_score(Y_test, LR_pred,average = 'weighted')) 
print("Recall:",metrics.recall_score(Y_test, LR_pred,average = 'weighted')) 


# In[462]:


#Display Naive Bayes metrics
print("Accuracy:",metrics.accuracy_score(Y_test, NB_pred)) 
print("Precision:",metrics.precision_score(Y_test, NB_pred,average='weighted')) 
print("Recall:",metrics.recall_score(Y_test, NB_pred,average = 'weighted')) 


# In[463]:


#Display SVM metrics
print("Accuracy:",metrics.accuracy_score(Y_test, SVM_pred)) 
print("Precision:",metrics.precision_score(Y_test, SVM_pred,average='weighted')) 
print("Recall:",metrics.recall_score(Y_test, SVM_pred,average = 'weighted')) 


# In[464]:


#Plot results of Logistic regression model
class_names=[0,1,2,3,4,5,6,7]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(LR_cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix for Logistic Regression - Amenities Score', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[465]:


#Plot results of Naive Bayes model
class_names=[0,1,2,3,4,5,6,7]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(NB_cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix for Logistic Regression - Amenities Score', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[466]:


#Plot results of SVM model
class_names=[0,1,2,3,4,5,6,7]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(SVM_cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix for SVM - Amenities Score', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[ ]:





# In[467]:


#Logistic Regression, Naive Bayes, SVM models for prediction of 'Access to Employment Score'
#Separate data into input and output
X = Grid_QOL.iloc[:,[0,1,3,4,5]].values
Y = Grid_QOL.iloc[:, 2].values


# In[468]:


#Split into training and test data
rand = np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = rand)


# In[469]:


#Standardize training/test data
sc_X = StandardScaler()
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[470]:


#Create Logistic Regression object
LR = LogisticRegression(max_iter=5000, random_state=0)
LR.fit(X_train,Y_train)


# In[471]:


#Create Naive Bayes object
NB = GaussianNB()
NB.fit(X_train, Y_train)


# In[472]:


#Create SVM Object
SVM = SVC()
SVM.fit(X_train,Y_train)


# In[473]:


#Get predited values from model
LR_pred2 = LR.predict(X_test)


# In[474]:


#Get predicted values from Naive Bayes model
NB_pred2 = NB.predict(X_test)


# In[475]:


#Get predicted values from SVM model
SVM_pred2 = SVM.predict(X_test)


# In[476]:


#Create confusion matrix for results
LR_cnf_matrix2 = confusion_matrix(Y_test, LR_pred2)


# In[477]:


#Create confusion matrix for Naive Bayes results
NB_cnf_matrix2 = confusion_matrix(Y_test, NB_pred2)


# In[478]:


#Create confusion matrix for SVM results
SVM_cnf_matrix2 = confusion_matrix(Y_test, SVM_pred2)


# In[479]:


#Display logistic regression metrics
print("Accuracy:",metrics.accuracy_score(Y_test, LR_pred2)) 
print("Precision:",metrics.precision_score(Y_test, LR_pred2,average = 'weighted')) 
print("Recall:",metrics.recall_score(Y_test, LR_pred2,average = 'weighted')) 


# In[480]:


#Display Naive Bayes metrics
print("Accuracy:",metrics.accuracy_score(Y_test, NB_pred2)) 
print("Precision:",metrics.precision_score(Y_test, NB_pred2,average='weighted')) 
print("Recall:",metrics.recall_score(Y_test, NB_pred2,average = 'weighted')) 


# In[481]:


#Display SVM metrics
print("Accuracy:",metrics.accuracy_score(Y_test, SVM_pred2)) 
print("Precision:",metrics.precision_score(Y_test, SVM_pred2,average='weighted')) 
print("Recall:",metrics.recall_score(Y_test, SVM_pred2,average = 'weighted')) 


# In[482]:


#Plot results of Logistic regression model
class_names=[0,1,2,3,4,5]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(LR_cnf_matrix2), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix for Logistic Regression - Employment Score', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[483]:


#Plot results of Naive Bayes model
class_names=[0,1,2,3,4,5]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(NB_cnf_matrix2), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix for Logistic Regression - Employment Score', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[484]:


#Plot results of SVM model
class_names=[0,1,2,3,4,5]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(SVM_cnf_matrix2), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix for SVM - Employment Score', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[ ]:





# In[485]:


#Logistic Regression, Naive Bayes, SVM models for prediction of 'Environmental Justice Score'
#Separate data into input and output
X = Grid_QOL.iloc[:,[0,1,2,4,5]].values
Y = Grid_QOL.iloc[:, 3].values


# In[486]:


#Split into training and test data
rand = np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = rand)


# In[487]:


#Standardize training/test data
sc_X = StandardScaler()
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[488]:


#Create Logistic Regression object
LR = LogisticRegression(max_iter=5000, random_state=0)
LR.fit(X_train,Y_train)


# In[489]:


#Create Naive Bayes object
NB = GaussianNB()
NB.fit(X_train, Y_train)


# In[490]:


#Create SVM Object
SVM = SVC()
SVM.fit(X_train,Y_train)


# In[491]:


#Get predited values from model
LR_pred3 = LR.predict(X_test)


# In[492]:


#Get predicted values from Naive Bayes model
NB_pred3 = NB.predict(X_test)


# In[493]:


#Get predicted values from SVM model
SVM_pred3 = SVM.predict(X_test)


# In[494]:


#Create confusion matrix for results
LR_cnf_matrix3 = confusion_matrix(Y_test, LR_pred3)


# In[495]:


#Create confusion matrix for Naive Bayes results
NB_cnf_matrix3 = confusion_matrix(Y_test, NB_pred3)


# In[496]:


#Create confusion matrix for SVM results
SVM_cnf_matrix3 = confusion_matrix(Y_test, SVM_pred3)


# In[497]:


#Display logistic regression metrics
print("Accuracy:",metrics.accuracy_score(Y_test, LR_pred3)) 
print("Precision:",metrics.precision_score(Y_test, LR_pred3,average = 'weighted')) 
print("Recall:",metrics.recall_score(Y_test, LR_pred3,average = 'weighted')) 


# In[498]:


#Display Naive Bayes metrics
print("Accuracy:",metrics.accuracy_score(Y_test, NB_pred3)) 
print("Precision:",metrics.precision_score(Y_test, NB_pred3,average='weighted')) 
print("Recall:",metrics.recall_score(Y_test, NB_pred3,average = 'weighted')) 


# In[499]:


#Display SVM metrics
print("Accuracy:",metrics.accuracy_score(Y_test, SVM_pred3)) 
print("Precision:",metrics.precision_score(Y_test, SVM_pred3,average='weighted')) 
print("Recall:",metrics.recall_score(Y_test, SVM_pred3,average = 'weighted')) 


# In[500]:


#Plot results of Logistic regression model
class_names=[0,1,2,3,4,5]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(LR_cnf_matrix3), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix for Logistic Regression - Environmental Justice Score', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[501]:


#Plot results of Naive Bayes model
class_names=[0,1,2,3,4,5]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(NB_cnf_matrix3), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix for Logistic Regression - Environmental Justice Score', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[502]:


#Plot results of SVM model
class_names=[0,1,2,3,4,5]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(SVM_cnf_matrix3), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix for SVM - Environmental Justice Score', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[ ]:





# In[503]:


#Logistic Regression, Naive Bayes, SVM models for prediction of 'Access to Housing Score'
#Separate data into input and output
X = Grid_QOL.iloc[:,[0,1,2,3,5]].values
Y = Grid_QOL.iloc[:, 4].values


# In[504]:


#Split into training and test data
rand = np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = rand)


# In[505]:


#Standardize training/test data
sc_X = StandardScaler()
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[506]:


#Create Logistic Regression object
LR = LogisticRegression(max_iter=5000, random_state=0)
LR.fit(X_train,Y_train)


# In[507]:


#Create Naive Bayes object
NB = GaussianNB()
NB.fit(X_train, Y_train)


# In[508]:


#Create SVM Object
SVM = SVC()
SVM.fit(X_train,Y_train)


# In[509]:


#Get predited values from Logistic Regression model
LR_pred4 = LR.predict(X_test)


# In[510]:


#Get predicted values from Naive Bayes model
NB_pred4 = NB.predict(X_test)


# In[511]:


#Get predicted values from SVM model
SVM_pred4 = SVM.predict(X_test)


# In[512]:


#Create confusion matrix for Logistic Regression results
LR_cnf_matrix4 = confusion_matrix(Y_test, LR_pred4)


# In[513]:


#Create confusion matrix for Naive Bayes results
NB_cnf_matrix4 = confusion_matrix(Y_test, NB_pred4)


# In[514]:


#Create confusion matrix for SVM results
SVM_cnf_matrix4 = confusion_matrix(Y_test, SVM_pred4)


# In[515]:


#Display logistic regression metrics
print("Accuracy:",metrics.accuracy_score(Y_test, LR_pred4)) 
print("Precision:",metrics.precision_score(Y_test, LR_pred4,average='weighted')) 
print("Recall:",metrics.recall_score(Y_test, LR_pred4,average = 'weighted')) 


# In[516]:


#Display Naive Bayes metrics
print("Accuracy:",metrics.accuracy_score(Y_test, NB_pred4)) 
print("Precision:",metrics.precision_score(Y_test, NB_pred4,average='weighted')) 
print("Recall:",metrics.recall_score(Y_test, NB_pred4,average = 'weighted')) 


# In[517]:


#Display SVM metrics
print("Accuracy:",metrics.accuracy_score(Y_test, SVM_pred4)) 
print("Precision:",metrics.precision_score(Y_test, SVM_pred4,average='weighted')) 
print("Recall:",metrics.recall_score(Y_test, SVM_pred4,average = 'weighted')) 


# In[518]:


#Plot results of Logistic regression model
class_names=[0,1,2,3,4,5,6,7]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(LR_cnf_matrix4), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix for Logistic Regression - Housing Score', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[519]:


#Plot results of Naive Bayes model
class_names=[0,1,2,3,4,5,6,7]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(NB_cnf_matrix4), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix for Naive Bayes - Housing Score', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[520]:


#Plot results of SVM model
class_names=[0,1,2,3,4,5,6,7]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

#Create heatmap 
sns.heatmap(pd.DataFrame(SVM_cnf_matrix4), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix for SVM - Housing Score', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




