#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()
os.chdir('C:\\Users\\meric\Desktop\codes')


# In[12]:


import pandas as pd
df=pd.read_csv('Advertising.csv')
# df=df.iloc[:,1:len(df)]
df.head()
df.info()


# In[17]:


import seaborn as sns
sns.jointplot(x="TV",y="Sales",data=df,kind="reg")


# In[21]:


from sklearn.linear_model import LinearRegression
# X=df[["TV"]]
# X.head()
# Y=df[["Sales"]]
# Y.head()


# In[24]:


reg=LinearRegression()
model=reg.fit(X,Y)
# dir(model)

# B1 SABIT KATSAYI
model.intercept_ 
# B2,B3 VS..
model.coef_


# In[25]:


# rkare
model.score(X,Y)


# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt
# ci=none güven aralıkları koyma anlamında
g=sns.regplot(df["TV"],df["Sales"],ci=None,scatter_kws={'color':'r','s':9})
g.set_title("modeel denklemi")
g.set_ylabel("satıs")
g.set_xlabel("tv harcamaları")
plt.xlim(-10,310)
plt.ylim(bottom=0)


# In[28]:


# y=ax+b de x yerine 165 konulunca predict edilen y değerini döndürür
model.predict([[165]]) 


# In[30]:


yeni_veri=[[5],[15],[30]]
model.predict(yeni_veri)


# In[50]:


gercek_y=Y[0:10]
# gercek_y
tahmin_edilen_y_array=model.predict(X)[0:10]
tahmin_edilen_y=pd.DataFrame(tahmin_edilen_y_array)
# tahmin_edilen_y
hatalar=pd.concat([gercek_y,tahmin_edilen_y],axis=1)
hatalar
# isimlendirme:
hatalar.columns=["gercek y","tahmin"]
hatalar


# In[52]:


hatalar["hata"]=hatalar["gercek y"]-hatalar["tahmin"]
hatalar


# In[54]:


hatalar["hata_karaler"]=hatalar["hata"]**2
hatalar


# In[56]:


import numpy as np
np.mean(hatalar["hata_karaler"])

