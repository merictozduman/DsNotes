#!/usr/bin/env python
# coding: utf-8

# # APRIORI ALGORİTMASI KULLANARAK BİRLİKTELİK KURAL ANALİZİ 
# 
# X: 1.ürün Y: 2.ürün N: Toplam Alışveriş
# 
# - Destek (Support) : Ürünlerin Birlikte Geçme Olasılığı
# 
# Support(X, Y) = Freq(X,Y)/N
# 
# ( X ve Y farklı ürünlerinin birlikte geçme frekansı / Toplam Alışveriş )
# 
# Support değerini hesaplama amacımız eşik değeri belirlemek. Veri setine bakıldığında bir çok ürün beraber görülebilir eşik değerinin altında kalan ürünlerde eleme yapacağız.
# 
# - Güven (Confidence) : X’i Alanların Y’yi Alma Olasılığı
# 
# Confidence(X, Y) = Freq(X,Y)/Freq(X)
# 
# ( X ve Y farklı ürünlerinin birlikte geçme frekansı / X’in gözlenme frekansı )
# 
# Analiz yapıldığında ister support değerine göre ister confidence değerine göre eşik değeri belirlenebilir. Eşik değerini gerçek hayat verilerinde çok az alabiliriz.
# 
# - Lift = Support(X,Y)/(Support(X)*Support(Y))
# 
# X ürünü alanların Y ürünü satın alması şu kadar kat artıyor yorumu vardır.

# # Data Understanding

# In[3]:


# .csv formatındaki veri setini okuma işlemi
# MILK,BREAD,BISCUIT olan sutün başlığını names=['products'] ile products'a çevirdik.
import os
import pandas as pd

os.chdir('C:\\Users\\meric\Desktop\codes')

df = pd.read_csv("datasets_344_727_GroceryStoreDataSet.csv", names=['products'], header = None)
df.head()


# In[4]:


# 20 gözlem ve 1 sütun var.

df.shape


# In[5]:


# Sutün adı ve tipi.

df.columns


# In[4]:


# Her satırda gözlenen gözlemler

df.values


# In[5]:


# Her satırda tek bir gözlem birimi varmış gibiydi bunu virgül ile tek tek ayırma işlemi yaptık.
data = list(df["products"].apply(lambda x:x.split(',')))
data 


# # Data Preprocessing

# In[7]:


#!pip install mlxtend


# In[ ]:


from mlxtend.preprocessing import TransactionEncoder


# In[8]:


#Veri setini istenilen True-False array'ine çevirdik.
# transactionEncoder için detay : http://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/

te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data,columns=te.columns_)
df


# # Data Association Rules

# In[10]:


from mlxtend.frequent_patterns import apriori


# In[11]:


# Apriori Fonksiyonunu deneyelim.
# df = True-False array ya da One Hot Encoding ile dönüştürülmüş dataframe
# min_support = Tüm kombinasyonların support değerini istemiyorum bu yüzden belirli bir eşik değerini getir.
# use_colnames = True, sutün isimlerini göster.
# verbose = 1, Toplam kombinasyon sayısını verir.

df1 = apriori(df, min_support=0.02, use_colnames=True, verbose = 1)
df1


# In[12]:


''' BISCUIT tüm alışverişlerin % 35'inde, 
    BREAD tüm alışverişlerin %65'inde veya 
    TEA, MAGGI, BREAD, BISCUIT tüm satışların % 5'inde beraber yorumları yapılır.''' 


# In[13]:


#Alışverişlerde en çok alınan ürünleri yorumlayabilmek için veriyi büyükten küçüğe sıralayabiliriz.

df1.sort_values(by="support", ascending=False)


# In[14]:


from mlxtend.frequent_patterns import association_rules


# In[15]:


# Aprioride "support" hesabı yapabiliriz, "confidence ve diğerleri" için association rules kullanıyoruz.

association_rules(df1, metric = "confidence", min_threshold = 0.6)


# In[16]:


''' antecedent support ; Birincinin tek başına görülme olasılığı,
    consequent support ; İkincinin tek başına görülme olasılığı,
    support ; İkisinin birlikte görülme olasılığı,
    confidence ; İlki satıldığında ikinci ürünün satılma olasılığı,
    lift ;  İlki satıldığında ikinci ürünün satılma olasılığı şu kadar kat arttı yorumu

'''


# ## Selecting and Filtering Results 

# In[17]:


''' "confidence" göz önünde bulundurulması gereken tek metrik değil. 
Bir kaç metriğin kombinasyonuda alınabilir.
'''

# http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

# linkten aldığım bir iki  acıklamalar :
# Leverage computes the difference between the observed frequency of A 
# and C appearing together and the frequency that would be expected 
# if A and C were independent. An leverage value of 0 indicates independence.



# şurdaki açıklamalar da fena değil :
# https://michael.hahsler.net/research/recommender/associationrules.html    


# In[36]:


rules = association_rules(df1, metric = "confidence", min_threshold = 0.6)
rules[ (rules['confidence'] >= 0.6) & (rules['support'] >= 0.2) ]


# # Reporting

# İndirgediğim veriseti üzerinden analiz yapıyor olacağım.
# - Alışverişlerde MAGGI'ın tek başına görülme olasığı %25, TEA'in tek başına görülme olasılığı %35.
# - 100 alışverişin 20'sinde mutlaka MAGGI ve TEA beraber satın alınıyor.
# - MAGGI satıldığında TEA satılma olasılığı 0.800.. yani %80.
# - MAGGI satılan satışlarda TEA satılma olasılığı 2.28 kat artmaktadır.
# 
# 
# Aksiyon Fikri: 
# - MAGGI alan biri %80 gibi yüksek bir ihtimalle TEA almaktadır ve TEA satışını 2.28 artırmaktadır. Bu iki ürün birbirinden uzak yerlerde konumlandırılarak müşterinin market içi dolaşması sağlanabilir, bu süreçte müşteri diğer ürünlere göz atabilir ve yahut satın alabilir. 
# 

# In[ ]:




