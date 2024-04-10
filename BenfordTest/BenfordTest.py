#!/usr/bin/env python
# coding: utf-8

# In[24]:


import benford as bf
from IPython import get_ipython


# In[25]:

#jupyter notebook'u python file'a çevirince aşağıdaki satır otomatik geliyor. Bunu kapatmak lazım.
#get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import pandas as pd

os.chdir('C:\\Users\\meric\Desktop\codes')


# In[26]:


sp = pd.read_csv('SPY.csv', index_col='Date', parse_dates=True)


# In[27]:


#adding '_' to facilitate handling the column
sp.rename(columns={'Adj Close':'Adj_Close'}, inplace=True) 
sp['p_r'] = sp.Close/sp.Close.shift()-1        #simple returns
sp['l_r'] = np.log(sp.Close/sp.Close.shift())  #log returns
# sp.tail()


# In[28]:


print(bf.first_digits(sp.l_r, digs=1, decimals=8),file=open("OutputBenfordtxt.txt", "a")) 
bf.first_digits(sp.l_r, digs=1, decimals=8,show_plot=True,save_plot='C:\\Users\meric\Desktop\codes\OutputBenford.pdf') 

