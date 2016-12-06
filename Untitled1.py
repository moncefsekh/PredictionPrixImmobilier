
# coding: utf-8

# ## Importation de Graphlab , et exploration des données

# In[1]:

import graphlab


# In[2]:

ventes = graphlab.SFrame('C:/home_data.csv')


# In[3]:

ventes


# ## Filtration des données

# In[4]:

ventes.remove_columns(['sqft_lot15','sqft_living15','condition','view','grade'])


# ## visualisation des données

# In[5]:

graphlab.canvas.set_target('ipynb')
ventes.show(view ="Scatter Plot", x="sqft_living", y="price")


# ## Séparation des données en TrainSet , TestSet

# In[6]:

train_set,test_set = ventes.random_split(.8,seed=0)  ## Train_set 80% et Test_set %20


# ## Création de quelques combinaisons de propriétés 

# In[7]:

ensemble_1 =['bedrooms','floors','bathrooms','yr_built','sqft_living','sqft_lot','zipcode','yr_renovated']
ensemble_2 =['bedrooms','floors','bathrooms','yr_built','sqft_living','zipcode','yr_renovated']
ensemble_3 =['bedrooms','sqft_lot','bathrooms','yr_built','zipcode','yr_renovated']
ensemble_4 =['bedrooms','bathrooms','sqft_living','zipcode','waterfront','lat','long']
ensemble_5 =['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
ensemble_6 =['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode','waterfront','yr_built','yr_renovated'
              ,'sqft_basement','sqft_above']


# ## Création des modéles de régression 

# In[8]:

model_1 = graphlab.linear_regression.create(train_set,target='price',features= ensemble_1)
model_2 = graphlab.linear_regression.create(train_set,target='price',features= ensemble_2)
model_3 = graphlab.linear_regression.create(train_set,target='price',features= ensemble_3)
model_4 = graphlab.linear_regression.create(train_set,target='price',features= ensemble_4)
model_5 = graphlab.linear_regression.create(train_set,target='price',features= ensemble_5)
model_6 = graphlab.linear_regression.create(train_set,target='price',features= ensemble_6)


# ## Evaluation des modèles  (RMSE)

# In[18]:

print model_1.evaluate(test_set)
print model_2.evaluate(test_set)
print model_3.evaluate(test_set)
print model_4.evaluate(test_set)
print model_5.evaluate(test_set)
print model_6.evaluate(test_set)


# In[19]:

test_set


# ## Prise des 3 premières maisons du TestSet pour tester 

# In[20]:

Maison_1 = ventes[ventes['id'] == 114101516]  
Maison_2 = ventes[ventes['id'] == 9297300055]
Maison_3 = ventes[ventes['id'] == 1202000200]


# In[21]:

print Maison_1


# ## Application des modèles

# In[22]:

model_1.predict(Maison_1)


# In[23]:

model_2.predict(Maison_1)


# In[24]:

model_3.predict(Maison_1)


# In[25]:

model_4.predict(Maison_1)


# In[26]:

model_5.predict(Maison_1)


# In[27]:

model_6.predict(Maison_1)


# In[28]:

print Maison_2


# ## Application des modèles

# In[29]:

model_1.predict(Maison_2)


# In[30]:

model_2.predict(Maison_2)


# In[31]:

model_3.predict(Maison_2)


# In[32]:

model_4.predict(Maison_2)


# In[33]:

model_5.predict(Maison_2)


# In[34]:

model_6.predict(Maison_2)


# In[35]:

print Maison_3


# ## Application des modèles

# In[36]:

model_1.predict(Maison_3)


# In[37]:

model_2.predict(Maison_3)


# In[38]:

model_3.predict(Maison_3)


# In[39]:

model_4.predict(Maison_3)


# In[40]:

model_5.predict(Maison_3)


# In[41]:

model_6.predict(Maison_3)


# In[25]:

House = [{'zipcode':98101, 
          'sqft_living':1722,
          'waterfront':1,
          'bedrooms':2,
          'bathrooms':1,
          'yr_built':2008
         }]


# In[38]:

model_4.predict(House)


# In[71]:

House2 = [{'zipcode':98103, 
          'sqft_living':5145,
          'waterfront':0,
          'bedrooms':4,
          'bathrooms':3,
          'yr_built':2016,
           'lat':47.683808,
           'long':-122.357354,
         }]


# In[72]:

model_4.predict(House2)


# In[ ]:



