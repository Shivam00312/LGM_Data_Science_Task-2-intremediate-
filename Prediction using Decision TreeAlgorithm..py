#!/usr/bin/env python
# coding: utf-8

# # LGM DATASCIENCE | AUGUST 2021
# 
# Author: Rashi Gupta
# 
# Level: Intermediate
# 
# Task-2: Prediction using Decision Tree  Algorithm
# 
# Dataset link: https://drive.google.com/file/d/11Iq7YvbWZbt8VXjfm06brx66b10YiwK-/view

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv('Iris.csv')
data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[8]:


data.shape


# In[11]:


data.columns


# In[14]:


data.Species.unique()


# In[15]:


data.isnull().sum()


# In[16]:


data.corr()


# In[19]:


sns.heatmap(data.corr(), cmap="twilight_r")


# In[20]:


sns.pairplot(data, hue='Species', vars=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])


# In[21]:


sns.heatmap(data.corr(),annot=True)


# In[22]:


sns.catplot(x='Species',y='PetalLengthCm',data=data)


# In[25]:


x=data.iloc[:,[1,2,3,4,]].values
y=data.iloc[:,-1].values


# In[28]:


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
y_new=ohe.fit_transform(data[['Species']])
y_new


# In[30]:


df=pd.DataFrame(y_new)
df.rename(columns={0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}, inplace=True)
df.head()


# In[32]:


result = pd.concat([data,df], axis=1)
result.head()


# In[33]:


result


# In[34]:


new_x=result.iloc[:,[1,2,3,4]].values
new_y=result.iloc[:,[5,6,7]].values


# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[36]:


train_x,test_x,train_y,test_y = train_test_split(new_x,new_y,test_size=0.2,random_state=1)


# In[37]:


print("Shape of train_x is ()".format(train_x.shape))
print("Shape of train_y is ()".format(train_y.shape))
print("Shape of test_x is ()".format(test_x.shape))
print("Shape of test_y is ()".format(test_y.shape))


# In[38]:


pip install dtc


# In[41]:


data = data.copy()
x=data.iloc[:,1:4]
y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=43)


# In[42]:


print(x_train.shape)


# In[43]:


print(y_train.shape)


# In[44]:


print(y_test.shape)


# In[45]:


print(x_test.shape)


# # Decision Tree Algorithm

# In[46]:


from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)


# In[47]:


print(classification_report(y_test,y_pred))


# In[48]:


print("Training score: ",clf.score(x_train,y_train))


# In[49]:


print(accuracy_score(y_test,y_pred))


# In[50]:


print(confusion_matrix(y_test,y_pred))


# In[53]:


data={'y_Actual':y_test,
     'y_Predicted':y_pred
     }
df=pd.DataFrame(data)
df.reset_index(inplace=True, drop=True)
df.head()


# In[55]:


pip install clf


# In[56]:


pred=clf.predict(x_test) #to make prediction on the test dataset
print(pred)


# In[57]:


print(clf.score(x_test,y_test))


# In[ ]:


# Importing libraries in Python
import sklearn.datasets as datasets
import pandas as pd

# Loading the iris dataset
iris=datasets.load_iris()

# Forming the iris dataframe
df=pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head(5))

y=iris.target
print(y)


# ### Now let us define the Decision Tree Algorithm

# In[ ]:


# Defining the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(df,y)

print('Decision Tree Classifer Created')


# ### Let us visualize the Decision Tree to understand it better.
# 
# 

# In[ ]:


# Install required libraries
get_ipython().system('pip install pydotplus')
get_ipython().system('apt-get install graphviz -y')


# In[ ]:


# Import necessary libraries for graph viz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# Visualize the graph
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names=iris.feature_names,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

