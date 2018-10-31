#coding:utf-8
# 2018.10.30
# mason_P first nn design for classify

import time,timeit
import logging
import os

TM=time.strftime("%Y:%m:%d-%H:%M",time.localtime())

LOG_FORMAT="%(asctime)s - %(levelname)s - [%(filename)s,line:%(lineno)d] - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def func_track(func):
    def track(*args, **kwargs):
        name=func.__name__
        logging.info("into the func %s..."%name)
        result=func(*args, **kwargs)
        logging.info("...func %s out"%name)
        return result
    return track


def mkdir(path):
    folder=os.path.exists(path)
    if not folder:
        os.makedirs(path)
        logging.info("--- create folder: %s"%path)
    else:
        logging.info("folder is exists")
    return 0



device="/cpu:0"

mkdir("./checkpoints/")
#checkpoint_dir= "./checkpoints/"
checkpoint_save_path="./checkpoints/model.ckp"
iter_counter_path="./checkpoints/model.ckp.iter"
final_model_path="./checkpoints/final_model/"
train_logdir="./tensorboard/train/"
test_logdir="./tensorboard/test/"

#params
lr=0.001 #learning rate
epoch_n=100
batch_size=8
#batch_n= int(np.ceil(len(data) / batch_size))
show_iter=100
checkpoint_iter=10000
#classes= len(label)# the labels, classes=y_train or y_test

lambd=0.01# 正则系数
