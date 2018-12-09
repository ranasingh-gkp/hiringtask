
# coding: utf-8

# In[67]:


import numpy as np
import tensorflow as tf
import random as rn
import os
from keras.layers import Dense, Convolution2D, Dropout, LSTM,GRU, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector, Merge
from keras.models import Sequential, Model,load_model
from keras.optimizers import Nadam, RMSprop
from keras import regularizers
from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import datetime

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

dataset = dataset[['date','views','subscriber','videoscount','year','day_of_week','month','day','week_number','season','quarter','part_of_month']].values.tolist()

# Splitting the dataset into train and test set

timesteps = 7
train_len = int(1*len(dataset)/timesteps)*timesteps # make it a multiple of timesteps
total_len = int((int(len(dataset))/timesteps)*timesteps)

train_y,train_x = list(), list()
test_y,test_x = list(), list()

for i in range(train_len):
    train_y.append(dataset[i][1:4])
    train_x.append(dataset[i][1:])
    
for i in range(train_len,total_len):
    test_y.append(dataset[i][1:4])
    test_x.append(dataset[i][1:])


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



input_train, output_train = list(), list()

for i in range(0,len(train_x)-timesteps):
    input_train.append(train_x[i:i+timesteps])
    output_train.append(train_y[i+timesteps])

input_np = np.array(input_train)
output_np = np.array(output_train)

input_train = input_np.reshape(input_np.shape[0],timesteps,input_np.shape[2])
output_train = output_np.reshape(output_np.shape[0],3,)

model = load_model(output_model_path)

final_out = [['date','views','subscriber','videoscount']]

data = []

for i in range(train_len-7,train_len):
	data.append(dataset[i])

for i in range(0,91):
	inp = []
	dt = ""
	ll = []
	dtx = []
	#print len(data)
	for j in range (i,i+7):
		inp.append(data[j][1:])
		dt = data[j][0]

	for ii in range(len(inp)):
	    for jj in range(len(inp[ii])):
	        inp[ii][jj] = (float(inp[ii][jj])-minmax_x[jj][0])/(minmax_x[jj][1]-minmax_x[jj][0])

	inp = np.array(inp)
	inp = inp.reshape(1,inp.shape[0],inp.shape[1])
	prediction = model.predict(inp)


	dt = datetime.datetime.strptime(dt, "%Y-%m-%d" ).date()
	dt += datetime.timedelta(days=1)

	ll.append(dt.strftime("%Y-%m-%d"))

	prediction = prediction[0]
	ll.append(int((prediction[0]*(minmax_x[0][1]-minmax_x[0][0]))+minmax_x[0][0]))
	ll.append(int((prediction[1]*(minmax_x[1][1]-minmax_x[1][0]))+minmax_x[1][0]))
	ll.append(int((prediction[2]*(minmax_x[2][1]-minmax_x[2][0]))+minmax_x[2][0]))

	dtx.append(dt.strftime("%Y-%m-%d"))
	dtx.append(int((prediction[0]*(minmax_x[0][1]-minmax_x[0][0]))+minmax_x[0][0]))
	dtx.append(int((prediction[1]*(minmax_x[1][1]-minmax_x[1][0]))+minmax_x[1][0]))
	dtx.append(int((prediction[2]*(minmax_x[2][1]-minmax_x[2][0]))+minmax_x[2][0]))

	final_out.append(dtx)

	ll.append(dt.year)
	ll.append(dt.weekday())
	ll.append(dt.month)
	ll.append(dt.day)
	ll.append(dt.strftime('%V'))
	seasons = [0,0,1,1,1,2,2,2,3,3,3,0]
	ll.append(seasons[dt.month-1])
	quarters = [0,0,0,1,1,1,2,2,2,3,3,3] #quarter
	ll.append(quarters[dt.month-1])
	part_of_months = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2] #quarter
	ll.append(part_of_months[dt.day-1])
	data.append(ll)


df = pd.DataFrame(final_out)
df.to_csv("output_3_months.csv")
