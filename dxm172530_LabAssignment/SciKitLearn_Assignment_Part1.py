
# coding: utf-8

# In[4]:


x=[[0. ,0.],[0. ,1.],[1. ,0.],[1. ,1.]]
y=[0,1,1,0]
from sklearn.neural_network import MLPClassifier


# In[5]:


clf = MLPClassifier(solver='sgd', activation='logistic', alpha=1e-5, random_state=1, hidden_layer_sizes=(3,2),max_iter=10000 )


# In[6]:


clf.fit(x, y)


# In[8]:


clf.predict([[2., 2.], [-1., -2.]])


# In[9]:


clf.predict([[2., 2.], [-1., -2.], [2., 2.], [-1., -2.]])


# In[10]:


clf.predict([[2., 2.], [-1., -2.], [2., 2.], [-1., -1.]])


# In[11]:


clf.predict([[2., 2.], [2., 2.], [2., 2.], [-1., -1.]])


# In[12]:


clf.coefs_[0]


# In[13]:


clf.intercepts_[0]

