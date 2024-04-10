#!/usr/bin/env python
# coding: utf-8

# In[70]:


import os
import pandas as pd
import matplotlib
import numpy as np
import scipy.stats as st

#os.getcwd()
#os.chdir('C:\\Users\\meric\Desktop\codes')
#os.listdir()
#dataset=pd.read_csv('USD_TRY.csv')

dataset=pd.read_excel('USD_TRY.xlsx',sheet_name='Sheet1')




dataset['Fark'].plot.hist(figsize=(8,8),color="green",bins=120)



# dataset.describe()

# dataset2=dataset[dataset["Fark"]<0.15]

# dataset.dtypes

# dataset.info()

# dataset.describe()

# dataset

# dataset[dataset["Tarih"]<'2017-01-04']

#dataset2=dataset[dataset["Tarih"]<'2017-01-04']

# dataset2

################################################################

def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


get_best_distribution(dataset['Fark'])


# In[ ]:




