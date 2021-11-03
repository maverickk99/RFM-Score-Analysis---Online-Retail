#!/usr/bin/env python
# coding: utf-8

# In[2]:


#-----DATA UNDERSTANDING-----

#installation of libraries
import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt

#to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);

#we determined how many numbers to show after comma
pd.set_option('display.float_format', lambda x: '%.0f' % x)
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv('C:\\Users\Hakan\Downloads\marketing.csv', encoding = 'unicode_escape')
data.head()


# In[4]:


#ranking of the most ordered products
data.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()


# In[5]:


#how many invoices are there in the data set
data["InvoiceNo"].nunique()


# In[6]:


#which are the most expensive products?
data.sort_values("UnitPrice", ascending = False).head()


# In[7]:


#top 5 countries with the highest number of orders
data["Country"].value_counts().head()


# In[8]:


#total spending was added as a column
data['TotalPrice'] = data['UnitPrice']*data['Quantity']


# In[9]:


data.head()


# In[10]:


#-----DATA PREPARATION-----
data["InvoiceDate"].min() #oldest shopping date


# In[11]:


data["InvoiceDate"].max() #newest shopping date


# In[12]:


#to make the assessment easier, today's date is set as January 1, 2012.  
today = pd.datetime(2012,1,1) 
today


# In[13]:


#changing the data type of the order date
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])


# In[14]:


#taking values greater than 0, this will be easier in terms of evaluation
data = data[data['Quantity'] > 0]
data = data[data['TotalPrice'] > 0]


# In[15]:


data.shape #size information


# In[16]:


data.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T
#explanatory statistics values of the observation units corresponding to the specified percentages
#processing according to numerical variables


# In[17]:


#-----Finding RFM Score-----#


# In[18]:


data.info()


# In[19]:


# finding Recency and Monetary values.
data_x = data.groupby('CustomerID').agg({'TotalPrice': lambda x: x.sum(), #monetary value
                                        'InvoiceDate': lambda x: (today - x.max()).days}) #recency value
#x.max()).days; last shopping date of customers


# In[20]:


data_y = data.groupby(['CustomerID','InvoiceDate']).agg({'TotalPrice': lambda x: x.sum()})
data_z = data_y.groupby('CustomerID').agg({'TotalPrice': lambda x: len(x)}) 
#finding the frequency value per capita


# In[21]:


rfm_table= pd.merge(data_x,data_z, on='CustomerID')
#creating the RFM table


# In[22]:


#determination of column names
rfm_table.rename(columns= {'InvoiceDate': 'Recency',
                          'TotalPrice_y': 'Frequency',
                          'TotalPrice_x': 'Monetary'}, inplace= True)


# In[23]:


rfm_table.head()


# In[24]:


#RFM score values 
rfm_table['RecencyScore'] = pd.qcut(rfm_table['Recency'],5,labels=[5,4,3,2,1])
rfm_table['FrequencyScore'] = pd.qcut(rfm_table['Frequency'].rank(method="first"),5,labels=[1,2,3,4,5])
rfm_table['MonetaryScore'] = pd.qcut(rfm_table['Monetary'],5,labels=[1,2,3,4,5])


# In[25]:


rfm_table.head()


# In[26]:


#RFM score values are combined side by side in str format
(rfm_table['RecencyScore'].astype(str) + 
 rfm_table['FrequencyScore'].astype(str) + 
 rfm_table['MonetaryScore'].astype(str)).head()


# In[27]:


#calculation of the RFM score
rfm_table["RFM_SCORE"] = rfm_table['RecencyScore'].astype(str) + rfm_table['FrequencyScore'].astype(str) + rfm_table['MonetaryScore'].astype(str)


# In[28]:


rfm_table.head()


# In[29]:


#transposition of the RFM table. This makes it easier to evaluate.
rfm_table.describe().T


# In[30]:


#customers with RFM Score 555
rfm_table[rfm_table["RFM_SCORE"] == "555"].head()


# In[31]:


#customers with RFM Score 111
rfm_table[rfm_table["RFM_SCORE"] == "111"].head()


# In[32]:


#segmenting of customers according to RecencyScore and FrequencyScore values
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Lose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}


# In[33]:


#creation of segment variable
rfm_table['Segment'] = rfm_table['RecencyScore'].astype(str) + rfm_table['FrequencyScore'].astype(str)
rfm_table['Segment'] = rfm_table['Segment'].replace(seg_map, regex=True)


# In[34]:


rfm_table.head()


# In[35]:


rfm_table[["Segment", "Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"])


# In[ ]:




