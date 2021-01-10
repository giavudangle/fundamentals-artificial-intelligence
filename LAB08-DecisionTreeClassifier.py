#!/usr/bin/env python
# coding: utf-8

# In[10]:


# b8_p1_1
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 

col_names = ['pregant','glucose','bp','skin','insulin','bmi','pedigree','age','label'
           ]
#load dataset
pima = pd.read_csv("D:\\diabetes.csv",header=0,names = col_names)
pima.head()

#split
feature_cols=['pregant','glucose','bp','skin','insulin','bmi','pedigree','age']
X = pima[feature_cols] #features
y = pima.label #target variable

print(pima[feature_cols])
print(pima.label)
#split ,train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
# create decision tree classifer object
clf = DecisionTreeClassifier()
#train
clf = clf.fit(X_train,y_train)
#predict
y_pred = clf.predict(X_test)
print(y_pred)
#model Accuracy
print("Accuracy",metrics.accuracy_score(y_test,y_pred))


# In[18]:


#b8_p1_2
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn import preprocessing

col_names =['gender','carownership','travelcost','incomelevel','transportationmode']
#load dataset
pima = pd.read_csv("D:\\transport.csv",header = 0,names =col_names)
pima.head()

lE = preprocessing.LabelEncoder()
data = pima.apply(lE .fit_transform)
print(data)

#split
feature_cols = ['gender','carownership','travelcost','incomelevel']
X = data[feature_cols]
y =data.transportationmode

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

#predict
y_pred = clf.predict(X_test)
print(y_pred)

print("Accuracy:" ,metrics.accuracy_score(y_test,y_pred))

d1 = np.array([[0,2,1,2]])
d1_pred = clf.predict(d1)
print(d1_pred)
ketqua=lE.inverse_transform(d1_pred)
print(ketqua)


# In[20]:


#b8_p2_1
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
import sklearn.metrics as metric
import numpy as np
X_training = [  [1,1],
                [1,0],  
                [0,1],
                [0,0],
             ]
y_training = [1,
              1,
              1,
              0
             ]
X_testing = X_training
y_true = y_training
ptn = Perceptron(max_iter= 500)
ptn.fit(X_training ,y_training)
y_pred=ptn.predict(X_testing)
print(y_pred)
accuracy = metric.accuracy_score(y_true,y_pred,normalize = True)
print('accuracy = ',accuracy)


# In[23]:


#b8_p2_2
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
import sklearn.metrics as metric
import numpy as np
X_training = [  [1,1],
                [1,0],  
                [0,1],
                [0,0],
             ]
y_training = [1,
              1,
              1,
              0
             ]
X_testing = X_training
y_true = y_training


mlp =  MLPClassifier(solver = 'lbfgs',hidden_layer_sizes=[1,1],activation='logistic')
mlp.fit(X_training,y_training)

y_pred=mlp.predict(X_testing)
print(y_pred)
accuracy = metric.accuracy_score(np.array(y_true).flatten(),np.array(y_pred).flatten(),normalize = True)
print('accuracy = ',accuracy)


# In[26]:


#b8_p2_3
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
import sklearn.metrics as metric
import numpy as np
X_training = [  [ 1,  1,  0],
                [ 1, -1, -1],  
                [-1,  1,  1],
                [-1, -1,  1],
                [ 0,  1, -1],
                [ 0, -1, -1],
                [ 1,  1,  1]
             ]
y_training = [[1,0],
              [0,1],
              [1,1],
              [1,0],
              [1,0],
              [1,1],
              [1,1]
              
             ]
X_testing = X_training
y_true = y_training


mlp =  MLPClassifier(solver = 'lbfgs',hidden_layer_sizes=[3,2],activation='logistic')
mlp.fit(X_training,y_training)

y_pred=mlp.predict(X_testing)
print(y_pred)
accuracy = metric.accuracy_score(np.array(y_true).flatten(),np.array(y_pred).flatten(),normalize = True)
print('accuracy = ',accuracy)


# In[4]:


#b8_p2_b4
import pandas as pd
import numpy as np
import sklearn.metrics as metric
from sklearn import metrics 
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

col_names = ['Outlook','Temp','Humidity','Windy','Play']
dt = pd.read_csv("D:\\Play.csv",header = 0 , names = col_names)
dt.head()

lE = preprocessing.LabelEncoder()
data = dt.apply(lE.fit_transform)
print (data)

feature_cols = ['Outlook','Temp','Humidity','Windy']

X = data[feature_cols]
y = data.Play

X_train,X_test,y_train,y_test  = train_test_split(X ,y ,test_size = 0.3 , random_state = 1 )

mlp =  MLPClassifier(solver = 'lbfgs',hidden_layer_sizes=[3,2],activation='logistic')
mlp.fit(X_train,y_train)

y_pred=mlp.predict(X_test)
print(y_pred)

print("Accuracy:" ,metrics.accuracy_score(y_test,y_pred))

d1 = np.array([[2,0,0,1]])
d1_pred = mlp.predict(d1)
print(d1_pred)
ketqua=lE.inverse_transform(d1_pred)
print(ketqua)


# In[ ]:




