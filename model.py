
# coding: utf-8

# In[67]:


import numpy as np
import tensorflow as tf
import random as rn
import os
from keras.layers import Dense, Convolution2D, Dropout, LSTM,GRU, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector, Merge
from keras.models import Sequential, Model
from keras.optimizers import Nadam, RMSprop
from keras import regularizers
from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
        
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)



output_model_path = "model_weights.h5"


# Load the dataset
dataset = read_csv('data.csv', index_col=None)

dataset = dataset[['views','subscriber','videoscount','year','day_of_week','month','day','week_number','season','quarter','part_of_month']].values.tolist()

# Splitting the dataset into train and test set

timesteps = 7
train_len = int(1*len(dataset)/timesteps)*timesteps # make it a multiple of timesteps
total_len = int((int(len(dataset))/timesteps)*timesteps)

train_y,train_x = list(), list()
test_y,test_x = list(), list()

for i in range(train_len):
    train_y.append(dataset[i][0:3])
    train_x.append(dataset[i][:])
    
for i in range(train_len,total_len):
    test_y.append(dataset[i][0:3])
    test_x.append(dataset[i][:])


# Manually normalizing and scaling the data

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

minmax_x = dataset_minmax(train_x)

min_y = min(train_y)
max_y = max(train_y)

norm = list()


# Scaling the training features
for i in range(len(train_x)):
    for j in range(len(train_x[i])):
        train_x[i][j] = (float(train_x[i][j])-minmax_x[j][0])/(minmax_x[j][1]-minmax_x[j][0])
    
# Scaling the training true output
for i in range(len(train_y)):
    for j in range(len(train_y[i])):
        train_y[i][j] = (float(train_y[i][j])-minmax_x[j][0])/(minmax_x[j][1]-minmax_x[j][0])
# Scaling the testing features
for i in range(len(test_x)):
    for j in range(len(test_x[i])):
        test_x[i][j] = (float(test_x[i][j])-minmax_x[j][0])/(float(minmax_x[j][1])-minmax_x[j][0])
    
# Scaling the testing true output
for i in range(len(test_y)):
    for j in range(len(test_y[i])):
        test_y[i][j] = (float(test_y[i][j])-minmax_x[j][0])/(float(minmax_x[j][1])-minmax_x[j][0])


input_train, output_train = list(), list()

for i in range(0,len(train_x)-timesteps):
    input_train.append(train_x[i:i+timesteps])
    output_train.append(train_y[i+timesteps])

input_np = np.array(input_train)
output_np = np.array(output_train)

input_train = input_np.reshape(input_np.shape[0],timesteps,input_np.shape[2])
output_train = output_np.reshape(output_np.shape[0],3,)


# Definfing the model

feature_expansion = 32
model = Sequential()
model.add(Dense(feature_expansion, input_shape=(input_train.shape[1], input_train.shape[2]),kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l1(0.01)))
model.add(LSTM(64, input_shape=(input_train.shape[1], feature_expansion)))
model.add(Dense(3, input_dim=timesteps,kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l1(0.01)))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# Training the model
for i in range(0,40):

    history = model.fit(input_train, output_train, epochs=1, batch_size=512, verbose=0, shuffle=False)

    #prediction = model.predict(input_train)

    print ("epoch",i),"loss=",history.history["loss"]

        
model.save(output_model_path)

print "training complete.."