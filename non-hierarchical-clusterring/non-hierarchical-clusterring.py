#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os

os.chdir('C:\\Users\\meric\Desktop\codes')

# read and query data
file = open("Clustering_gmm.csv")
data = pd.read_csv(file)
print("query data")
print(f"{data.head()} \n {data.info()}")



# In[2]:


# plot data
plt.scatter(x = data["Weight"], y = data["Height"])
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()


# In[6]:


# we see 4 clusters in the data
n_cl = 4
# one cluster is distant from the other 3
# on fitting a model, we should get 4 clearly defined clusters

# fit kmeans model
km = KMeans(n_clusters = n_cl)
km.fit(data)
# predict cluster for each observation
km_cl = km.predict(data)
# append cluster to each observation
data["km_cluster"] = km_cl
print(f"kmeans clustering \n {data.head()}")
# plot kmeans clusters
km_col = ["red", "green", "blue", "cyan"]
for i in range(n_cl):
    obs = data[data["km_cluster"] == i]
    plt.scatter(x = obs["Weight"], y = obs["Height"], c = km_col[i])
    plt.xlabel("Weight")
    plt.ylabel("Height")
plt.show()


# In[10]:


# fit GMM model
gmm = GaussianMixture(n_components = n_cl)
gmm.fit(data.loc[:, ["Weight", "Height"]])
# predict cluster for each observation
gmm_cl = gmm.predict(data.loc[:, ["Weight", "Height"]])
# append cluster to each observation
data["gmm_cluster"] = gmm_cl
print(f"gmm clustering \n {data.head()}")
# plot gmm clusters
gmm_col = ["cyan", "green", "orange", "purple"]
for i in range(n_cl):
    obs = data[data["gmm_cluster"] == i]
    plt.scatter(x = obs["Weight"], y = obs["Height"], c = gmm_col[i])
    plt.xlabel("Weight")
    plt.ylabel("Height")
plt.show()


# In[12]:


# fit DBSCAN model
dbs = DBSCAN(eps = 0.9, min_samples = 5)
# predict cluster for each observation
dbs_cl = dbs.fit_predict(data.loc[:, ["Weight", "Height"]])
# append cluster to each observation
data["dbs_cluster"] = dbs_cl
print(f"dbscan clustering \n {data.head()}")
# plot dbscan clusters
dbs_col = ["black", "blue", "green", "purple"]
for i in range(n_cl):
    obs = data[data["dbs_cluster"] == i]
    plt.scatter(x = obs["Weight"], y = obs["Height"], c = dbs_col[i])
    plt.xlabel("Weight")
    plt.ylabel("Height")
plt.show()


# In[13]:


# fit BIRCH model
brc = Birch(threshold = 0.05, n_clusters = 4)
brc.fit(data.loc[:, ["Weight", "Height"]])
# predict cluster for each observation
brc_cl = brc.predict(data.loc[:, ["Weight", "Height"]])
# append cluster to each observation
data["brc_cluster"] = brc_cl
print(f"birch clustering \n {data.head()}")
# plot birch clusters
brc_col = ["cyan", "green", "orange", "red"]
for i in range(n_cl):
    obs = data[data["brc_cluster"] == i]
    plt.scatter(x = obs["Weight"], y = obs["Height"], c = brc_col[i])
    plt.xlabel("Weight")
    plt.ylabel("Height")
plt.show()

