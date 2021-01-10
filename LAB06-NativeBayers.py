#!/usr/bin/env python
# coding: utf-8

# In[2]:


#bài tập 1
from __future__ import print_function
from sklearn.naive_bayes import MultinomialNB
import numpy as np
# train data
d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]
train_data = np.array([d1, d2, d3, d4])
label = np.array(['B', 'B', 'B', 'N'])
# test data
d5 = np.array([[2, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])
## call MultinomialNB
clf = MultinomialNB()
# training
clf.fit(train_data, label)
# test
print('Predicting class of d5:', str(clf.predict(d5)[0]))
print('Probability of d6 in each class:', clf.predict_proba(d6))


# In[3]:


#bài 2
from __future__ import print_function
from sklearn.naive_bayes import BernoulliNB
import numpy as np
# train data
d1 = [1, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]
train_data = np.array([d1, d2, d3, d4])
label = np.array(['B', 'B', 'B', 'N']) # 0 - B, 1 - N
# test data
d5 = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])
## call BernoulliNB
clf = BernoulliNB()
# training
clf.fit(train_data, label)
# test
print('Predicting class of d5:', str(clf.predict(d5)[0]))
print('Probability of d6 in each class:', clf.predict_proba(d6))


# In[4]:


#Bài 3

from __future__ import print_function
from sklearn.naive_bayes import MultinomialNB
import numpy as np 

#train data
d1 = [0, 0, 0, 1]
d2 = [0, 0, 0, 0]
d3 = [1, 0, 0, 1]
d4 = [2, 1, 0, 1]
d5 = [2, 2, 1, 1]
d6 = [2, 2, 1, 0]
d7 = [1, 2, 1, 0]
d8 = [0, 1, 0, 1]
d9 = [0, 2, 1, 1]
d10 = [2, 1, 1, 1]
d11 = [0, 1, 1, 0]
d12 = [1, 1, 0, 0]
d13 = [1, 0, 1, 1]
d14 = [2, 1, 0, 0]

train_data = np.array([d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14])
label = np.array(['No', 'No', 'Yes', 'Yes', 'Yes','No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No' ]) 

d15 = np.array([[0, 1, 0, 0]])
## call MultinomialNB
clf = MultinomialNB()
# training 
clf.fit(train_data, label)

# test
print('Predicting class of d5:', str(clf.predict(d15)[0]))
print('Probability of d6 in each class:', clf.predict_proba(d15))


# In[ ]:




