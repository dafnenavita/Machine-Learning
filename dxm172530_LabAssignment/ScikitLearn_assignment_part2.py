
# coding: utf-8

# In[15]:


import pandas as pd
wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names = ["Cultivar", "Alchol", "Malic", "Ash", "Alcalinity", "Magnesium", "Total", "Falvanoids", "Nonflavanoid", "Proanthocyanins", "Color", "Hue", "OD", "Proline"])


# In[16]:


wine.describe()


# In[17]:


wine.shape


# In[19]:


wine.head()


# In[20]:


# Classify the data 
X = wine.drop('Cultivar',axis=1)
y = wine['Cultivar']


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


# split the data into test and train 


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[24]:


#data preprocessing 
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
 


# In[26]:


scale.fit(X_train)


# In[28]:


X_train = scale.transform(X_train)
X_test = scale.transform(X_test)


# In[29]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(11,13,11),max_iter=500)
mlp.fit(X_train,y_train)


# In[30]:


clf = mlp.predict(X_test)


# In[31]:


from sklearn.metrics import classification_report,confusion_matrix


# In[33]:


print(confusion_matrix(y_test,clf))


# In[34]:


print(classification_report(y_test,clf))

