#coding:utf-8
# 2018.10.30
# mason_P first nn design for classify

import numpy as np
from  config import *
from sklearn.datasets import make_moons


###load dataset;
@classmethod
@func_track
def load_date():
    x_moons, y_moons=make_moons(m=2000,noise=0.1,random_state=42)
    return x_moons, y_moons

data,label=load_date()
classes= len(label)
batch_n= int(np.ceil(len(data) / batch_size))

###data separate
test_ratio = 0.2
test_size = int(len(data) * test_ratio)
X_train = data[:-test_size]
X_test = data[-test_size:]
y_train = label[:-test_size]
y_test = label[-test_size:]


### pick batch
@classmethod
@func_track
def random_batch(x_,y_, size):
    random_indices=np.random.randint(0,len(x_),size)#random
    x_batch=x_[random_indices]
    y_batch=y_[random_indices]
    return x_batch, y_batch

