#!/usr/bin/env python
# coding: utf-8

# # **Project idea** – Fake news spreads like a wildfire and this is a big issue in this era

# #  **Libraries** Are Used

# In[78]:


import numpy as np
import pandas as pd
import itertools
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[3]:


import nltk
nltk.download('stopwords')


# #**Printing** - Stopwords(En)

# In[4]:


print(stopwords.words('english'))


# # Dataset **Reading**

# In[5]:


data=pd.read_csv('news.csv')
data


# # Dataset - *Preprocessing and Clean*

# In[6]:


data.columns


# In[7]:


data.size


# In[8]:


data.describe


# In[9]:


data.shape


# In[10]:


data.info


# Drop Column(unrequired)

# In[11]:


data.drop("Unnamed: 0", axis=1, inplace=True)


# # Column null or Not

# In[12]:


data.isnull().sum()


# # Target Set

# In[13]:


target=data.label
target.head()


# # Dataset **Splitting**

# In[44]:


X_train,X_test,y_train,y_test=train_test_split(data['text'], target, test_size=0.2, random_state=8)


# # Initializing A **TfidfVectorizer**

# In[45]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.5)


# # Transform & Fitting **Train Set**, Transforming Test Set

# In[46]:


tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)


# # Initializing ***Passive Aggressive Classifier***

# In[79]:


PassiveAggressiveClassifier=PassiveAggressiveClassifier(max_iter=50)
PassiveAggressiveClassifier.fit(tfidf_train,y_train)


# In[50]:


port_stem = PorterStemmer()


# In[51]:


def stemming(title):
    stemmed_title = re.sub('[^a-zA-Z]',' ',title)
    stemmed_title = stemmed_title.lower()
    stemmed_title = stemmed_title.split()
    stemmed_title = [port_stem.stem(word) for word in stemmed_title if not word in stopwords.words('english')]
    stemmed_title = ' '.join(stemmed_title)
    return stemmed_title


# In[52]:


data['title'] = data['title'].apply(stemming)


# In[53]:


print(data['title'])


# In[54]:


X = data.drop(columns='label', axis=1)
Y = data['label']


# In[55]:


print(X)


# In[56]:


print(Y)


# In[57]:


X = data['title'].values
Y = data['label'].values


# In[58]:


print(X)
print(Y)


# In[59]:


vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[60]:


print(X)
print(Y)


# # Accuracy **Score** Over Training Data

# In[65]:


X_train_predict = model.predict(tfidf_train)
training_accuracy = accuracy_score(X_train_predict, y_train)


# In[66]:


print('Accuracy score of the training data : ', training_accuracy)


# # Accuracy **Score** Over The Testing Data

# In[69]:


x_test_predict = model.predict(tfidf_test)
test_accuracy = accuracy_score(x_test_predict, y_test)


# In[70]:


print('Accuracy score of the test data : ', test_accuracy)


# In[41]:


model = LogisticRegression()


# In[63]:


model.fit(tfidf_train, y_train)


# # *Testing Of The* **Data**

# In[71]:


def fake_news_detection(news):
    input_data=[news]
    vectorized_input_data=tfidf_vectorizer.transform(input_data)
    predict=PassiveAggressiveClassifier.predict(vectorized_input_data)
    print(predict)
    


# In[72]:


fake_news_detection('You Can Smell Hillary’s Fear')


# In[88]:


X_new = tfidf_test[8]

pred = model.predict(X_new)
print(pred)

if (pred[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')


# # Predict on the test set and calculate accuracy

# In[98]:


y_pred=PassiveAggressiveClassifier.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# # The accuracy of 93.37% with this model.

# In[99]:


confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# # In this Model ,we have 590 True Positives,585 True Negatives,44 false positives ,and 48 false negatives.
# 

# # IN SUMMARY, we learned to detect fake news with Python. We took a political dataset, implemented a TfidfVectorizer, initialized a PassiveAggressiveClassifier, and fitted our model. We ended up obtaining an accuracy of 93.37% in magnitude.Thats all.
# 

# # Thank You

# In[ ]:




