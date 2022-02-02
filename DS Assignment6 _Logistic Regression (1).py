#!/usr/bin/env python
# coding: utf-8

# ## LOGISTIC REGRESSION

# In[1]:


#Loading Data
import pandas as pd
bank_details = pd.read_csv('C:/Users/17pol/Downloads/bank-full.csv', sep = ';')
bank_details.head()


# In[2]:


#importing Logistic Regression Module
from sklearn.linear_model import LogisticRegression


# In[3]:


#Performing EDA
bank_details.head()
bank_details.shape #(45211, 17)
bank_details.info() #all non-null
bank_details[bank_details.duplicated()] #no duplicates present
bank_details['previous'].unique()
bank_details['poutcome'].unique()
bank_details['contact'].unique()


# In[4]:


df2 = bank_details.copy()


# In[5]:


bank_details.info()


# In[6]:


#checking for correlations among the input variables
#chisquare test of independence

pip install researchpy
data = pd.read_csv('C:/Users/17pol/Downloads/bank-full.csv', sep = ';')
data.info()

import researchpy as rp
crosstab, test_results, expected = rp.crosstab(data["job"], data["age"],
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")
crosstab
test_results
#Cramers V is >0.15 thus age and job have a strong correlation

crosstab, test_results,expected = rp.crosstab(data['housing'], data['loan'], 
                                              test = 'chi-square', expected_freqs = True,prop = 'cell')
crosstab
test_results
#Cramer's value is <0.05 thus very weak relation

crosstab, test_results, expected = rp.crosstab(data['age'],data['education'],
                                              test= 'chi-square', expected_freqs = True, prop = 'cell')
crosstab
test_results
#moderate to strong relation between age and education with cramer's V 0.15

crosstab, test_results, expected = rp.crosstab(data['previous'],data['campaign'],
                                              test= 'chi-square', expected_freqs = True, prop = 'cell')
crosstab
test_results
#no or very weak relation between previous and campaign

crosstab, test_results, expected = rp.crosstab(data['age'],data['marital'],
                                              test= 'chi-square', expected_freqs = True, prop = 'cell')
crosstab
test_results
#Very strong relation between age and marital

crosstab, test_results, expected = rp.crosstab(data['pdays'],data['previous'],
                                              test= 'chi-square', expected_freqs = True, prop = 'cell')
crosstab
test_results
#very strong relation between pdays and previous

crosstab, test_results, expected = rp.crosstab(data['housing'],data['balance'],
                                              test= 'chi-square', expected_freqs = True, prop = 'cell')
crosstab
test_results
#very strong relation between housing and balance


crosstab, test_results, expected = rp.crosstab(data['loan'],data['balance'],
                                              test= 'chi-square', expected_freqs = True, prop = 'cell')
crosstab
test_results
#very strong realtion between loan and balance

crosstab, test_results, expected = rp.crosstab(data['duration'],data['previous'],
                                              test= 'chi-square', expected_freqs = True, prop = 'cell')
crosstab
test_results
#moderate to strong relation between duration and previous


crosstab, test_results, expected = rp.crosstab(data['duration'],data['contact'],
                                              test= 'chi-square', expected_freqs = True, prop = 'cell')
crosstab
test_results
#strong relation between contact and duration


crosstab, test_results, expected = rp.crosstab(data['previous'],data['contact'],
                                              test= 'chi-square', expected_freqs = True, prop = 'cell')
crosstab
test_results
#strong relation between contact and previous


crosstab, test_results, expected = rp.crosstab(data['default'],data['balance'],
                                              test= 'chi-square', expected_freqs = True, prop = 'cell')
crosstab
test_results
#very strong relation between default and balance

crosstab, test_results, expected = rp.crosstab(data['campaign'], data['previous'],
                                              test = 'chi-square', expected_freqs = True, prop = 'cell' )
crosstab
test_results
#weak relation between campaign and previous

crosstab, test_results, expected = rp.crosstab(data['day'], data['y'],
                                              test = 'chi-square', expected_freqs = True, prop = 'cell' )
crosstab
test_results
#moderate relation

crosstab, test_results, expected = rp.crosstab(data['month'], data['y'],
                                              test = 'chi-square', expected_freqs = True, prop = 'cell' )
crosstab
test_results
#strong relation

crosstab, test_results, expected = rp.crosstab(data['poutcome'], data['previous'],
                                              test = 'chi-square', expected_freqs = True, prop = 'cell' )
crosstab
test_results
#very strog relation



# We keep just the following predictors considering the strong relation within different variables
# 1. age
# 2. education
# 3. previous
# 4. balance
# 5. campaign
# 6. day
# 7. month

# In[ ]:


data.columns


# In[ ]:


#CREATING A NEW DATAFRAME WITH THE REQUIRED INPUT VARIABLES
df1 = data[['age','education','balance','day','month','campaign','previous','y']]


# In[ ]:


df1['y'] = df1['y'].astype('category')
df1 = pd.get_dummies(df1, columns = ['y'])

df1['month'] = df1['month'].astype('category')
df1 = pd.get_dummies(df1, columns = ['month'])

df1['education'] = df1['education'].astype('category')
df1 = pd.get_dummies(df1, columns = ['education'])


# In[ ]:


#CONCATING DUMMIES CREATED FOR Y AS WE WANT JUST ONE OUTPUT COLUMN
df1['y'] = pd.concat([df1['y_no'], df1['y_yes']], ignore_index = True)
df1['y']


# In[7]:


#checking for inputs
df1.info()


# In[ ]:


#Assigning predictots to X and predictions to Y
X = bank_details.iloc[:,0:21]
Y = bank_details.iloc[:,23]


# In[ ]:


classifier = LogisticRegression()
classifier.fit(X,Y)


# In[ ]:


#Predict for X dataset
y_pred = classifier.predict(X)

y_pred_df= pd.DataFrame({'actual': Y,
                         'predicted_prob': classifier.predict(X)})
y_pred_df


# In[ ]:


# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)


# In[ ]:


#Sensitivity
s = 12500/(12500+4144)

#Specificity
d = 19058/(9509+19058)

#Precision
p = 12500/(12500+9509)

s,d

#good sensitivity and specificity A


# In[ ]:


#Classification report
from sklearn.metrics import classification_report
print(classification_report(Y,y_pred))


# In[ ]:


#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(Y, classifier.predict_proba (X)[:,1])

auc = roc_auc_score(Y, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='blue', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')

