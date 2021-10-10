#!/usr/bin/env python
# coding: utf-8

# # Market Basket Analysis

# In[1]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori


# In[2]:


data1=pd.read_csv("C:/Users/HP/Downloads/Market_Basket_Optimisation.csv",header=None)
print(data1)


# In[3]:


transaction=[]
for i in range(0,len(data1)):
    transaction.append([str(data1.values[i,j])for j in range(0,20) if str(data1.values[i,j])!='nan'])
print(transaction)


# In[4]:


t=TransactionEncoder()
array=t.fit(transaction).transform(transaction)
print(array)


# In[6]:


df=pd.DataFrame(array,columns=t.columns_)
print(df)


# In[7]:


frequent_itemsets = apriori(df,min_support=0.003, use_colnames=True)
print(frequent_itemsets)


# In[8]:


rules=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.5)
print(rules)


# In[15]:


print(rules[rules.conviction > 2.5])


# In[ ]:




