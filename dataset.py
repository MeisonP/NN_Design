#coding:utf-8
# 2018.10.30
# mason_P first nn design for classify

import numpy as np


###load dataset;
def load_date():
    return


###data seperite
test_size=

x_train=
y_train=

x_test=
y_test=


### pick batch
def random_batch(x_,y_,batch_size):
    random_indices=np.random.randint(0,len(x_),batch_size)#random
    x_batch=x_[random_indices]
    y_batch=y_[random_indices]
    return x_batch, y_batch

