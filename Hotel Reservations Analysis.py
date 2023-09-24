#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boto3
import pandas as pd
import numpy as np
from sagemaker import get_execution_role
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


# In[2]:


role = get_execution_role()
bucketname = 'fangfei-adhoc'
filename = 'Hotel Reservations.csv'


# In[3]:


data_input_location = 's3://{}/{}'.format(bucketname, filename)


# In[4]:


## read data
data_raw = pd.read_csv(data_input_location, sep=",")


# In[5]:


data_raw.head()


# ## Exploring Data Analysis

# In[6]:


data_raw.columns


# In[7]:


## Get a general info of the data; columns are non-null, so no missing value need fill in
data_raw.info()


# In[8]:


# check the target class distribution, find out the class is imbalanced distributed
data_raw.booking_status.value_counts()


# In[9]:


sns.set_style("darkgrid")
ax=sns.countplot(x=data_raw.booking_status)
ax.set_title('Target column: booking_status distribution')
plt.show()


# In[10]:


data_raw.describe().T


# In[11]:


data_raw.describe(include='all').T


# ## Feature Engineer

# In[12]:


## the booking_id has no impact to the target class, so remove it
data_raw=data_raw.drop(columns='Booking_ID')


# In[13]:


data_raw.info()


# In[14]:


data_raw[['type_of_meal_plan']]


# In[15]:


## As SVM, NN, KNN can't deal with category data, so do a one-hot encoding to transfer into binary columns:
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore',sparse=False)
data_tmp=pd.DataFrame(enc.fit_transform(data_raw[['type_of_meal_plan']])).rename(columns={0:'tmp0',1:'tmp1',2:'tmp2',3:'tmp3'})
data_tmp


# In[16]:


data_rtr=pd.DataFrame(enc.fit_transform(data_raw[['room_type_reserved']])).rename(columns={0:'rtr0',1:'rtr1',2:'rtr2',3:'rtr3',4:'rtr4',5:'rtr5',6:'rtr6'})
data_rtr.head()


# In[17]:


data_mst=pd.DataFrame(enc.fit_transform(data_raw[['market_segment_type']])).rename(columns={0:'mst0',1:'mst1',2:'mst2',3:'mst3',4:'mst4'})
data_mst


# In[18]:


data=data_raw.join(data_mst).join(data_rtr).join(data_tmp)
data.drop(columns=(['type_of_meal_plan','room_type_reserved','market_segment_type']),inplace=True)
data.head()


# ## Modelling

# ### Decision Tree

# In[23]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# In[24]:


data.columns


# In[25]:


Y=data['booking_status']
X=data.loc[:,data.columns!='booking_status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[26]:


clf=DecisionTreeClassifier()
clf=clf.fit(X_train,Y_train)
Y_pred_train=clf.predict(X_train)


# In[27]:


train_f1=metrics.f1_score(Y_train,Y_pred_train,average='micro')
train_f1


# In[28]:


Y_pred=clf.predict(X_test)
test_f1=metrics.f1_score(Y_test,Y_pred,average='micro')
test_f1


# In[29]:


## Since data is imbalanced, we will chose "macro" for f1_score which gives bigger penalty if model does not perform well for minority class
metrics.f1_score(Y_test,Y_pred,average='macro')


# In[30]:


## micro not faving in any class
metrics.f1_score(Y_test,Y_pred,average='micro')


# In[31]:


## wighted will give more weight for majority class which is not good for imbalanced data
metrics.f1_score(Y_test,Y_pred,average='weighted')


# #### parameters to explore

# In[384]:


train_scores=[]
test_scores=[]
for i in range(1, 21):
    model = DecisionTreeClassifier(max_depth=i)
    model.fit(X_train,Y_train)
    train_pred = model.predict(X_train)
    train_f1 = metrics.f1_score(Y_train,train_pred ,average='micro') 
    train_scores.append(train_f1)

    test_pred = model.predict(X_test)
    test_f1 = metrics.f1_score(Y_test,test_pred ,average='micro') 
    test_scores.append(test_f1)


# In[391]:


plt.plot(range(1,21), train_scores, '-o', label='Train')
plt.plot(range(1,21), test_scores, '-o', label='Test')
plt.title('F1 score for different max_depth parameter')
plt.xlabel('max_depth parameter')
plt.ylabel('F1 score')
plt.xticks(np.arange(0, 21, step=1))
plt.legend()
plt.show()


# In[398]:


train_scores=[]
test_scores=[]
values=range(2,15)
for i in values:
    model = DecisionTreeClassifier(min_samples_split=i)
    model.fit(X_train,Y_train)
    train_pred = model.predict(X_train)
    train_f1 = metrics.f1_score(Y_train,train_pred ,average='micro') 
    train_scores.append(train_f1)

    test_pred = model.predict(X_test)
    test_f1 = metrics.f1_score(Y_test,test_pred ,average='micro') 
    test_scores.append(test_f1)


# In[399]:


plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label='Test')
plt.title('F1 score for different min_samples_split parameter')
plt.xlabel('min_samples_split parameter')
plt.ylabel('F1 score')
plt.xticks(np.arange(2, 15, step=1))
plt.legend()
plt.show()


# #### GridSearch for the best parameters

# In[402]:


dt_param_grid = {"criterion": ['gini','entropy'],
              "max_depth": [None,5,10,15, 20],
              "min_samples_split": [0,2,3,5,20]
              }


# In[403]:


clf_GS = GridSearchCV(estimator=clf, param_grid=dt_param_grid,cv=5, scoring='f1_micro', return_train_score=True)
clf1=clf_GS.fit(X_train,Y_train)


# In[405]:


clf1.best_score_


# In[404]:


clf1.best_params_


# In[411]:


dc_gts=pd.DataFrame(clf1.cv_results_).reset_index()
dc_gts.head()


# In[419]:


plt.plot(dc_gts['index'], dc_gts['mean_train_score'], '-o', label='Train')
plt.plot(dc_gts['index'], dc_gts['mean_test_score'], '-o', label='Test')
plt.title('F1 score for different min_samples_split parameter')
plt.xlabel('min_samples_split parameter')
plt.ylabel('F1 score')
plt.xticks(np.arange(0,40, step=1))
plt.legend()
plt.show()


# In[60]:


## find the parameter
dt_param_grid = {"criterion": ['gini','entropy'],
              "max_depth": [None,5,10,13,15, 17, 20],
              "min_samples_split": [0,1,2,3,5,10]
              }


# In[61]:


clf_GS = GridSearchCV(estimator=clf, param_grid=dt_param_grid,cv=5, scoring='f1_micro', return_train_score=True)
clf1=clf_GS.fit(X_train,Y_train)


# In[62]:


clf1.best_score_


# In[63]:


clf1.best_params_


# ## Boosting

# #### Adaboost

# In[423]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier


# In[433]:


clf_boost = AdaBoostClassifier(n_estimators=100).fit(X_train, Y_train)
Y_pred_boost1=clf_boost.predict(X_test)


# In[435]:


metrics.f1_score(Y_test,Y_pred_boost1,average='micro')


# In[447]:


train_scores=[]
test_scores=[]
values=range(10,300,20)
for i in values:
    model = AdaBoostClassifier(n_estimators=i)
    model.fit(X_train,Y_train)
    train_pred = model.predict(X_train)
    train_f1 = metrics.f1_score(Y_train,train_pred ,average='micro') 
    train_scores.append(train_f1)

    test_pred = model.predict(X_test)
    test_f1 = metrics.f1_score(Y_test,test_pred ,average='micro') 
    test_scores.append(test_f1)


# In[444]:


min(values)


# In[448]:


plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label='Test')
plt.title('F1 score for different n_estimators parameter')
plt.xlabel('n_estimators parameter')
plt.ylabel('F1 score')
plt.xticks(np.arange(min(values), max(values), step=20))
plt.legend()
plt.show()


# In[452]:


train_scores=[]
test_scores=[]
values=[0.1,0.5,1,5,10]
for i in values:
    model = AdaBoostClassifier(learning_rate=i)
    model.fit(X_train,Y_train)
    train_pred = model.predict(X_train)
    train_f1 = metrics.f1_score(Y_train,train_pred ,average='micro') 
    train_scores.append(train_f1)

    test_pred = model.predict(X_test)
    test_f1 = metrics.f1_score(Y_test,test_pred ,average='micro') 
    test_scores.append(test_f1)


# In[454]:


plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label='Test')
plt.title('F1 score for different learning_rate parameter')
plt.xlabel('learning_rate parameter')
plt.ylabel('F1 score')
plt.xticks(np.arange(min(values), max(values), step=0.5))
plt.legend()
plt.show()


# In[468]:


boost_param= {
    "n_estimators": [50,100,150,200],
    "learning_rate":[0.1,0.5,1,5,10]
}


# In[469]:


clf_GS = GridSearchCV(estimator=clf_boost, param_grid=boost_param,cv=5, scoring='f1_micro', return_train_score=True)
clf_boost_GS=clf_GS.fit(X_train,Y_train)


# In[471]:


clf_boost_GS.best_params_


# In[474]:


Y_pred_boost3=clf_boost_GS.predict(X_test)


# In[475]:


metrics.f1_score(Y_test,Y_pred_boost3,average='micro')


# #### Gradient Boosting

# In[476]:


clf_boost2 = GradientBoostingClassifier().fit(X_train, Y_train)
Y_pred_boost2=clf_boost2.predict(X_test)


# In[478]:


metrics.f1_score(Y_test,Y_pred_boost2,average='micro')


# In[486]:


train_scores=[]
test_scores=[]
values=[0.1,0.5,1.0,5.0]
for i in values:
    model =  GradientBoostingClassifier(learning_rate=i).fit(X_train, Y_train)
    model.fit(X_train,Y_train)
    train_pred = model.predict(X_train)
    train_f1 = metrics.f1_score(Y_train,train_pred ,average='micro') 
    train_scores.append(train_f1)

    test_pred = model.predict(X_test)
    test_f1 = metrics.f1_score(Y_test,test_pred ,average='micro') 
    test_scores.append(test_f1)


# In[487]:


plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label='Test')
plt.title('F1 score for different learning_rate parameter')
plt.xlabel('learning_rate parameter')
plt.ylabel('F1 score')
plt.xticks(np.arange(min(values), max(values)))
plt.legend()
plt.show()


# In[491]:


train_scores=[]
test_scores=[]
values=[3,5,10,20,30,40]
for i in values:
    model =  GradientBoostingClassifier(max_depth=i).fit(X_train, Y_train)
    model.fit(X_train,Y_train)
    train_pred = model.predict(X_train)
    train_f1 = metrics.f1_score(Y_train,train_pred ,average='micro') 
    train_scores.append(train_f1)

    test_pred = model.predict(X_test)
    test_f1 = metrics.f1_score(Y_test,test_pred ,average='micro') 
    test_scores.append(test_f1)


# In[494]:


plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label='Test')
plt.title('F1 score for different max_depth parameter')
plt.xlabel('max_depth parameter')
plt.ylabel('F1 score')
plt.xticks(np.arange(min(values), max(values),10))
plt.legend()
plt.show()


# In[500]:


train_scores=[]
test_scores=[]
values=[100,150,200,250,300,350,400,450,500]
for i in values:
    model = GradientBoostingClassifier(n_estimators=i).fit(X_train, Y_train)
    model.fit(X_train,Y_train)
    train_pred = model.predict(X_train)
    train_f1 = metrics.f1_score(Y_train,train_pred ,average='micro') 
    train_scores.append(train_f1)

    test_pred = model.predict(X_test)
    test_f1 = metrics.f1_score(Y_test,test_pred ,average='micro') 
    test_scores.append(test_f1)


# In[501]:


plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label='Test')
plt.title('F1 score for different n_estimators parameter')
plt.xlabel('n_estimators parameter')
plt.ylabel('F1 score')
plt.xticks(np.arange(min(values), max(values),50))
plt.legend()
plt.show()


# In[502]:


## change the max_depth to 20, and learning rate to 0.01 and 10, best learning rate =1
clf_boost2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=20, random_state=0).fit(X_train, Y_train)


# In[503]:


Y_pred_boost2=clf_boost2.predict(X_test)


# In[504]:


metrics.f1_score(Y_test,Y_pred_boost2,average='micro')


# In[508]:


Gboost_param= {
    "n_estimators": [100,200,300,400],
    "learning_rate":[0.1,1,10],
    "max_depth":[10, 20, 30, 40]
}


# In[ ]:


clf_GS = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=Gboost_param,cv=5, scoring='f1_micro', return_train_score=True)
clf_boost_GS=clf_GS.fit(X_train, Y_train)


# In[505]:


clf_boost3 = GradientBoostingClassifier(n_estimators=450, learning_rate=1, max_depth=10, random_state=0).fit(X_train, Y_train)


# In[506]:


Y_pred_boost3=clf_boost3.predict(X_test)
metrics.f1_score(Y_test,Y_pred_boost3,average='micro')


# #### XGBoost

# In[88]:


get_ipython().system('/home/ec2-user/anaconda3/envs/python3/bin/python3 -m pip install  xgboost')


# In[90]:


from xgboost import XGBClassifier


# In[98]:


Y_train_boost3=np.where(Y_train=='Canceled',1,0)
Y_test_boost3=np.where(Y_test=='Canceled',1,0)


# In[96]:


clf_boost3 = XGBClassifier()
clf_boost3.fit(X_train, Y_train_boost3)


# In[119]:


Y_pred_boost3=clf_boost3.predict(X_test)


# In[127]:


metrics.f1_score(Y_pred_boost3,Y_test_boost3,average='micro')


# ### KNN
# knn need apply x and y in array format, so it needs train on X_train.values

# In[169]:


from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train.values, Y_train.values)


# In[170]:


y_pred_knn=knn.predict(X_test.values)


# In[171]:


metrics.f1_score(y_pred_knn,Y_test,average='micro')


# #### find the best parameter for KNN

# In[516]:


train_scores=[]
test_scores=[]
values=range(1,30)
for i in values:
    model =  KNeighborsClassifier(n_neighbors=i).fit(X_train.values, Y_train.values)
    train_pred = model.predict(X_train.values)
    train_f1 = metrics.f1_score(Y_train,train_pred ,average='micro') 
    train_scores.append(train_f1)
    test_pred = model.predict(X_test.values)
    test_f1 = metrics.f1_score(Y_test,test_pred ,average='micro') 
    test_scores.append(test_f1)


# In[518]:


plt.plot(values, test_scores, '-o', label='Test')
plt.title('F1 score for different n_neighbors parameter')
plt.xlabel('n_neighbors parameter')
plt.ylabel('F1 score')
plt.xticks(np.arange(min(values), max(values)))
plt.legend()
plt.show()


# In[517]:


plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label='Test')
plt.title('F1 score for different n_neighbors parameter')
plt.xlabel('n_neighbors parameter')
plt.ylabel('F1 score')
plt.xticks(np.arange(min(values), max(values)))
plt.legend()
plt.show()


# In[519]:


model =  KNeighborsClassifier(n_neighbors=5).fit(X_train.values, Y_train.values)
test_pred = model.predict(X_test.values)
metrics.f1_score(Y_test,test_pred ,average='micro')


# #### gridsearch

# In[173]:


parameters = {"n_neighbors": range(1, 10)}
gridsearch = GridSearchCV(KNeighborsClassifier(), parameters,scoring='f1_micro',return_train_score=True)


# In[174]:


gridsearch.fit(X_train.values, Y_train.values)


# In[175]:


gridsearch.best_params_["n_neighbors"]


# In[177]:


gridsearch.best_estimator_


# In[178]:


y_pred_knn_GS=gridsearch.predict(X_test.values)


# In[179]:


metrics.f1_score(y_pred_knn_GS,Y_test,average='micro')


# In[180]:


gridsearch.best_score_


# In[176]:


gridsearch.cv_results_


# In[203]:


plt.plot(range(1,10),gridsearch.cv_results_['mean_test_score'])


# ### SVM

# In[67]:


from sklearn import svm
svc =svm.SVC(kernel='linear') 


# In[68]:


svc.fit(X_train, Y_train)


# In[69]:


Y_pred_svc=svc.predict(X_test)
metrics.f1_score(Y_pred_svc,Y_test,average='micro')


# In[73]:


## use cross validation to check the data
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svc, X_train, Y_train, cv=5)


# In[74]:


scores


# In[75]:


parameters = {'kernel':('linear', 'rbf','sigmoid'), 
              'C':[0.1, 1, 10],'gamma': [1,0.1,0.01]}
svc_GS= GridSearchCV(svc, parameters)


# In[76]:


parameters = {'kernel':('linear', 'rbf','sigmoid'), 
              'C':[0.1, 1, 10]}
svc_GS= GridSearchCV(svc, parameters)


# In[77]:


svc_GS.fit(X_train, Y_train)


# In[78]:


svc_GS.best_params_


# In[83]:


y_pred_svc_GS=svc_GS.predict(X_test)


# In[84]:


metrics.f1_score(y_pred_svc_GS,Y_test,average='micro')


# In[212]:


svc_GS.cv_results_


# In[220]:


plt.plot(gridsearch.cv_results_['mean_test_score'])


# ### NN 
# keras

# In[20]:


get_ipython().system('pip install tensorflow')


# In[36]:


import tensorflow as tf


# In[32]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[33]:


## Need scale before NN
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  


# In[39]:


Y_train_boost3=np.where(Y_train=='Canceled',1,0)
Y_test_boost3=np.where(Y_test=='Canceled',1,0)


# In[34]:


model = Sequential()
model.add(Dense(12, input_shape=(30,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[41]:


model.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
model.fit(X_train, Y_train_boost3,epochs=10, batch_size=1, verbose=1)


# In[43]:


score = model.evaluate(X_test, Y_test_boost3,verbose=1)


# In[44]:





# In[45]:


y_pred_nn = model.predict(X_test)
y_pred_test_nn=np.where(y_pred_nn>0.5,1,0)
metrics.f1_score(y_pred_test_nn ,Y_test_boost3,average='micro')


# In[46]:


model = Sequential()
model.add(Dense(12, input_shape=(30,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
model.fit(X_train, Y_train_boost3,epochs=10, batch_size=1, verbose=1)


# In[48]:


y_pred_nn = model.predict(X_test)
y_pred_test_nn=np.where(y_pred_nn>0.5,1,0)
metrics.f1_score(y_pred_test_nn ,Y_test_boost3,average='micro')


# In[55]:


model = Sequential()
model.add(Dense(8, input_shape=(30,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[62]:


model.summary()


# In[58]:


model.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
model.fit(X_train, Y_train_boost3,epochs=10, batch_size=1, verbose=1)


# In[60]:


y_pred_nn = model.predict(X_test)
y_pred_test_nn=np.where(y_pred_nn>0.5,1,0)
metrics.f1_score(y_pred_test_nn ,Y_test_boost3,average='micro')


# In[ ]:





# In[52]:


model = Sequential()
model.add(Dense(8, input_shape=(30,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[53]:


model.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
model.fit(X_train, Y_train_boost3,epochs=5, batch_size=1, verbose=1)


# In[54]:


y_pred_nn = model.predict(X_test)
y_pred_test_nn=np.where(y_pred_nn>0.5,1,0)
metrics.f1_score(y_pred_test_nn ,Y_test_boost3,average='micro')


# In[63]:


model = Sequential()
model.add(Dense(12, input_shape=(30,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[64]:


model.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
model.fit(X_train, Y_train_boost3,epochs=10, batch_size=1, verbose=1)


# In[66]:


model.summary()


# In[65]:


y_pred_nn = model.predict(X_test)
y_pred_test_nn=np.where(y_pred_nn>0.5,1,0)
metrics.f1_score(y_pred_test_nn ,Y_test_boost3,average='micro')


# In[ ]:




