#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('car data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Fuel_Type'].unique())
print(df['Owner'].unique())


# In[6]:


#to check missing values
df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


final_datasets=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[10]:


final_datasets.head()


# In[11]:


final_datasets['current_year']=2021


# In[12]:


final_datasets.head()


# In[13]:


final_datasets['age_car']= final_datasets['current_year']-final_datasets['Year']


# In[14]:


final_datasets.head()


# In[15]:


final_datasets.drop(['current_year'],axis=1,inplace=True)


# In[16]:


final_datasets.head()


# In[17]:


final_datasets.drop(['Year'], axis=1, inplace=True)


# In[18]:


final_datasets


# In[19]:


final_datasets = pd.get_dummies(final_datasets, drop_first=True)


# In[20]:


final_datasets.head()


# In[21]:


final_datasets


# In[22]:


#find correlation
final_datasets.corr()


# In[23]:


import seaborn as sns


# In[24]:


sns.pairplot(final_datasets)


# In[25]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


corrmat=final_datasets.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_datasets[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[27]:


final_datasets.head()


# In[28]:


#selling price is dependent features and all others are independent feature
x=final_datasets.iloc[:,1:]
y= final_datasets.iloc[:,0]


# In[29]:


x.head()


# In[30]:


y.head()


# In[31]:


# feature_importances
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(x,y)


# In[32]:


print(model.feature_importances_)


# In[33]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(7).plot(kind='barh')
plt.show()


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[35]:


X_train.shape


# In[36]:


X_test.shape


# In[37]:


y_train.shape


# In[38]:


y_test.shape


# In[39]:


from sklearn.ensemble import RandomForestRegressor


# In[40]:


regressor=RandomForestRegressor()


# In[41]:


import numpy as np


# In[42]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[43]:


from sklearn.model_selection import RandomizedSearchCV


# In[44]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[45]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[46]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# In[47]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[48]:


rf_random.fit(X_train,y_train)


# In[49]:



rf_random.best_params_


# In[50]:


rf_random.best_score_


# In[51]:


predictions=rf_random.predict(X_test)


# In[52]:


sns.distplot(y_test-predictions)


# In[53]:



plt.scatter(y_test,predictions)


# In[54]:


from sklearn import metrics


# In[55]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[56]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[ ]:




