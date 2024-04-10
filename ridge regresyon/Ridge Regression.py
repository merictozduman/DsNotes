#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV


# In[1]:


import os
os.getcwd()
os.chdir('C:\\Users\\meric\Desktop\codes')


# In[2]:


# hem datayı aldık mı diye bakalım, hem içerik ne diye bakalım
import pandas as pd
df=pd.read_csv('Hitters.csv')
# df=df.iloc[:,1:len(df)]
df.head()
df.info()


# In[4]:


# içinde en az bir eksik veri olan satırları uçuruyoruz:
df=df.dropna()
df


# In[7]:


dms=pd.get_dummies(df[['League','Division','NewLeague']])
dms


# In[9]:


Y=df['Salary']
X_=df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
X=pd.concat([X_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)
X


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# In[19]:


ridge_model=Ridge(alpha=0.1).fit(X_train,Y_train)
ridge_model


# In[20]:


ridge_model.coef_
# değişik aplha değerleri için farklı coef'ler çıkacak.
#  deneme yanılma yöntemiyle optimum alpha ya karar verilecek


# In[21]:


ridge_model.intercept_


# In[23]:


# linspace(10,-2,100) 10 ile -2 arasında 100 değer
lambdalar=10**np.linspace(10,-2,100)*0.5
lambdalar


# In[26]:


ridge_model=Ridge()
katsayilar=[]

for i in lambdalar:
     ridge_model.set_params(alpha=i)
     ridge_model.fit(X_train,Y_train)
     katsayilar.append(ridge_model.coef_)   
        
katsayilar        


# In[27]:


ax=plt.gca()
ax.plot(lambdalar,katsayilar)
# katsayiların hepsini gözlemleyebilmek için rakamları birbirine 
# yakınlaştıralım deyip düzleştirme yapıyoruz
ax.set_xscale("log")
# renkler katsayı değerleri
# x ekseni lambda değerleri


# In[30]:


ridge_model=Ridge().fit(X_train,Y_train)
y_pred=ridge_model.predict(X_train)
y_pred[0:10]


# In[31]:


# train hatası
RMSE=np.sqrt(mean_squared_error(Y_train,y_pred))
RMSE


# In[35]:


from sklearn.model_selection import cross_val_score
np.sqrt(np.mean(-cross_val_score(ridge_model,X_train,Y_train,cv=10,scoring="neg_mean_squared_error")))


# In[38]:


# test hatası
y_pred=ridge_model.predict(X_test)
RMSE=np.sqrt(mean_squared_error(Y_test,y_pred))
RMSE
# test hatasını modeller arasında seçim yaparken
# train hatasını model oluştururken model parametreleri arasında 
# seçim yaparken kullanıyoruz


# In[41]:


ridge_model=Ridge(1).fit(X_train,Y_train)
y_pred=ridge_model.predict(X_test)
np.sqrt(mean_squared_error(Y_test,y_pred))


# In[44]:


# 0-100 arası 10 sayı
lambdalar1=np.random.randint(0,1000,100)
lambdalar2=10**np.linspace(10,-2,100)*0.5


# In[52]:


# cv: cross_Validation
# ridgecv=RidgeCV(alphas=lambdalar2,scoring='neg_mean_squared_error',cv=10,normalize=True)
ridgecv=RidgeCV(alphas=lambdalar1,scoring='neg_mean_squared_error',cv=10,normalize=True)
ridgecv.fit(X_train,Y_train)
ridgecv.alpha_


# In[49]:


#final
ridge_tuned=Ridge(alpha=ridgecv.alpha_).fit(X_train,Y_train)


# In[50]:


y_pred=ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(Y_test,y_pred))


# In[ ]:




