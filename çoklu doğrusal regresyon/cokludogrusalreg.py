#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
os.getcwd()
os.chdir('C:\\Users\\meric\Desktop\codes')


# In[4]:


import pandas as pd
df=pd.read_csv('Advertising.csv')
# df=df.iloc[:,1:len(df)]
df.head()
df.info()


# In[5]:


from sklearn.linear_model import LinearRegression
X=df[["TV","Radio","Newspaper"]]
# X.head()
Y=df[["Sales"]]
# Y.head()


# In[6]:


import statsmodels.api as sm

lm=sm.OLS(Y,X)
model=lm.fit()
model.summary()


# In[7]:


from sklearn.linear_model import LinearRegression
##scikit learn ile model kurmak
lm=LinearRegression()
model=lm.fit(X,Y)
model.intercept_
model.coef_


# In[8]:


yeni_veri=[[30],[10],[40]]
yeni_veri=pd.DataFrame(yeni_veri).T
yeni_veri


# In[9]:


model.predict(yeni_veri)


# In[10]:


from sklearn.metrics import mean_squared_error
# MSE
MSE=mean_squared_error(Y,model.predict(X))


# In[11]:


import numpy as np
RMSE=np.sqrt(MSE)
RMSE


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=99)
x_train
y_train
x_test
y_test.describe()


# In[24]:


# egitim hatası
lm=LinearRegression()
model=lm.fit(x_train,y_train)
np.sqrt(mean_squared_error(y_train,model.predict(x_train)))


# In[34]:


# test hatası
np.sqrt(mean_squared_error(y_test,model.predict(x_test)))


# In[35]:


# cvint, cross-validation generator or an iterable, default=None
from sklearn.model_selection import cross_val_score
cross_val_score(model,x_train,y_train,cv=10,scoring="neg_mean_squared_error")


# In[36]:


np.mean(-cross_val_score(model,x_train,y_train,cv=10,scoring="neg_mean_squared_error"))

