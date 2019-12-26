
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()  


# In[3]:


dataset.keys()


# In[4]:


cancer = pd.DataFrame(np.c_[dataset['data'],dataset['target']], 
             columns=np.append(dataset['feature_names'],['target']))


# In[5]:


cancer.head()


# In[6]:


cancer.columns


# In[7]:


sns.pairplot(cancer , vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness'],hue='target')


# In[8]:


sns.scatterplot(x='mean area',y='mean smoothness',data=cancer,hue='target')


# In[9]:


plt.figure(figsize=(20,10))
sns.heatmap(cancer.corr(),annot=True)


# In[10]:


X = cancer.drop('target',axis=1)
y = cancer['target']


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[12]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)


# In[13]:


from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)


# In[16]:


from sklearn.metrics import confusion_matrix,classification_report


# In[15]:


cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)


# In[18]:


print(classification_report(y_test,y_pred))

