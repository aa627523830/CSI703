#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

from functools import reduce
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from PIL import Image
import itertools
import io
import math
import plotly.offline as py
import plotly.offline as py # visualization
import plotly.graph_objs as go  # visualization
from plotly.subplots import make_subplots
import plotly.figure_factory as ff # visualization
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score  
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


from imblearn.under_sampling import RandomUnderSampler
import statsmodels.api as sm
from yellowbrick.classifier import DiscriminationThreshold

#plot in a map
import folium
from folium.plugins import MarkerCluster
from folium import Choropleth, Circle, Marker
from folium import plugins
from folium.plugins import HeatMap

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Import the file exported by mysql

df = pd.read_csv('Telco_Churn_from_mysql.csv')
df.head()


# # Dataframe size and info

# In[4]:


def get_df_size(df, header='Dataset dimensions'):
  print(header,
        '\n# Attributes: ', df.shape[1], 
        '\n# Entries: ', df.shape[0],'\n')
  
get_df_size(df)


# # Features and data types

# In[5]:


df.info()


# In[6]:


data = df[['Churn_Label','Country','State','City','Zip_Code','Latitude','Longitude']]


# In[7]:


data.head()


# In[8]:


def heatmap1(data):
    data['Latitude'] = data['Latitude'].astype(float)
    data['Longitude'] = data['Longitude'].astype(float)
    data_churn = data[data['Churn_Label'] == 'Yes']
    data_churn = data_churn[['Latitude', 'Longitude']]
    heat_data = [[row['Latitude'],row['Longitude']] for index, row in data_churn.iterrows()]
    mc = MarkerCluster()
    
    # Create the map
    map1 = folium.Map(location=[35.369587, -119.039496], zoom_start=6)
    for idx, row in data.iterrows():
        if not math.isnan(row['Longitude']) and not math.isnan(row['Latitude']):
            mc.add_child(Marker([row['Latitude'], row['Longitude']]))
    map1.add_child(mc)

    # Plot it on the map
    HeatMap(heat_data).add_to(map1)
    loc = 'Geographical distribution of churn customers'
    title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(loc)   
    map1.get_root().html.add_child(folium.Element(title_html))
    map1.save(outfile= "test.html")

    return map1


# In[9]:


heatmap1(data)


# # Unique values per feature

# In[10]:


print("Unique values (per feature): \n{}\n".format(df.nunique()))


# # Remove some arrtibutes

# In[11]:


coldrop=['Customer_ID','Count1','Under_30','Customer_ID.1','Count2','Country','State','City','Zip_Code','Latitude','Longitude'
         ,'Customer_ID.2','Count3','Quarter','Customer_ID.3','Count4','Quarter.1','Customer_Status','Churn_Value','Churn_Category'
         ,'Churn_Reason']
df=df.drop(coldrop, axis=1)
df.head()


# # Checking missing values

# In[12]:


num = df.isnull().sum()
print(num)


# In[13]:


# show rows with missing values
null_data = df[df.isnull().any(axis=1)]
null_data


# In[14]:


# Drop the rows where at least one element is missing.
# Before drop: Entries:  7043 
df = df.dropna()
get_df_size(df)


# # Split features into binary, numeric or categorical

# In[15]:


binary_feat = df.nunique()[df.nunique() == 2].keys().tolist()
numeric_feat = [col for col in df.select_dtypes(['float','int']).columns.to_list() if col not in binary_feat]
categorical_feat = [ col for col in df.select_dtypes('object').columns.to_list() if col not in binary_feat + numeric_feat ]


# # Apply label encoding for binary features

# In[16]:


le = LabelEncoder()
df_proc = df.copy()

for i in binary_feat:
  df_proc[i] = le.fit_transform(df_proc[i])
df_proc.head()


# # Convert categorical variable into dummy variables

# In[17]:


df_proc = pd.get_dummies(df_proc, columns=categorical_feat)


# In[18]:


df_proc.head()


# In[19]:


df_proc.info()


# # Radar chart

# In[20]:


tmp_churn = df_proc[df_proc['Churn_Label'] == 1]
tmp_no_churn = df_proc[df_proc['Churn_Label'] == 0]


# In[21]:


column = ['Gender','Married','Dependents','Phone_Service','Internet_Service','Paperless_Billing', 'Payment_Method_Credit Card','Contract_Month-to-Month']


# In[22]:


sub_df = df_proc[column]
sub_df.head()


# In[23]:


for i in column :
    tmp_churn[i] = le.fit_transform(tmp_churn[i])
for i in column :
    tmp_no_churn[i] = le.fit_transform(tmp_no_churn[i])


# In[24]:


df1 = tmp_churn[column].sum().reset_index()
df1.columns  = ['feature','Yes_n']
df1['No_n'] = tmp_churn.shape[0]  - df1['Yes_n']
df2 = tmp_no_churn[column].sum().reset_index()
df2.columns  = ['feature','Yes_n']
df2['No_n'] = tmp_churn.shape[0]  - df1['Yes_n']

df2.head()


# In[25]:


trace1 = go.Scatterpolar(r = df1['Yes_n'].values.tolist(), 
                         theta = df1['feature'].tolist(),
                         fill  = 'toself',name = 'Churn (Yes)',
                         mode = 'markers+lines', visible=True,
                         marker = dict(size = 5))
trace2 = go.Scatterpolar(r = df1['No_n'].values.tolist(), 
                         theta = df1['feature'].tolist(),
                         fill  = 'toself',name = 'Churn (No)',
                         mode = 'markers+lines', visible=True,
                         marker = dict(size = 5))
trace3 = go.Scatterpolar(r = df2['Yes_n'].values.tolist(), 
                         theta = df2['feature'].tolist(),
                         fill  = 'toself',name = 'NoChurn (Yes)',
                         mode = 'markers+lines', visible=True,
                         marker = dict(size = 5))
trace4 = go.Scatterpolar(r = df2['No_n'].values.tolist(), 
                         theta = df2['feature'].tolist(),
                         fill  = 'toself',name = 'NoChurn (No)',
                         mode = 'markers+lines', visible=True,
                         marker = dict(size = 5))


# In[26]:


plot = [trace1, trace2, trace3, trace4]
data = list([dict(active=0,x=-0.2,buttons=list([
    dict(label = 'Churn Customer',method = 'update',args = [{'visible': [True, True, False, False]}, {'title': 'Churn Customer Binary Features Counting Distribution'}]),
    dict(label = 'No-Churn Dist',method = 'update',args = [{'visible': [False, False, True, True]},{'title': 'No Churn Customer Binary Features Counting Distribution'}]),]),)])

layout = dict(showlegend=False,updatemenus=data)
fig = dict(data=plot, layout=layout)
iplot(fig)


# In[27]:


df_proc.head()


# In[31]:


sns.countplot(x="Churn_Label",hue="Gender", data=df_proc)
plt.title('Histogram for Gender')


# In[33]:


sns.countplot(x="Churn_Label",hue="Married", data=df_proc)
plt.title('Histogram for Married')


# In[ ]:


map1.save(outfile= "test.html")


# In[37]:


# show the distribution of numerical variable

def draw_subplots(var_Name,tittle_Name,nrow=1,ncol=1,idx=1,fz=10): # Define a common module for drawing subplots.
    ax = plt.subplot(nrow,ncol,idx)                   #  idx - position of subplot in the main plotting window
    ax.set_title('Distribution of '+var_Name)         #  fz - the font size of Tittle in the main plotting window

    
numeric_columns = ['Monthly_Charge','Total_Charges']

fig,ax = plt.subplots(1,1, figsize=(10,10))
fig.tight_layout()
j=0  # reset the counter to plot 
title_Str="Plotting the density distribution of various numeric variable"

for i in numeric_columns:
    j +=1
    draw_subplots(i,title_Str,2,1,j,20) # create a 1x2 subplots for plotting distribution plots
    sns.distplot(df[i])
    plt.xlabel('')


# In[43]:


sns.catplot(x="Multiple_Lines", y="Monthly_Charge", hue="Churn_Label", kind="violin",
                 split=True, palette="pastel", data=df_proc, height=4.2, aspect=1.4)
plt.title('Violin plot for Monthly_Charge')
plt.xticks([0, 1],['No','Yes'])


# In[44]:


sns.catplot(x="Gender", y="Monthly_Charge", hue="Churn_Label", kind="violin",
                 split=True, palette="pastel", data=df_proc, height=4.2, aspect=1.4)
plt.title('Violin plot for Monthly_Charge')
plt.xticks([0, 1],['No','Yes'])

