#coding:utf-8
# 2018.10.30
# mason_P first nn design for classify

import time,timeit
import logging
import os

TM=time.strftime("%Y:%m:%d-%H:%M:%S",time.localtime())

LOG_FORMAT="%(asctime)s - %(levelname)s - [line:%(lineno)d] - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def func_track(func):
    def track(*args, **kwargs):
        name=func.__name__
        logging.info("into the func %s..."%name)
        result=func(*args, **kwargs)
        logging.info("...func %s out"%name)
        return result
    return track

@func_track
def mkdir(path):
    folder=os.path.exists(path)
    if not folder:
        os.makedirs(path)
        logging.info("--- create folder: %s"%path)
    else:
        logging.info("folder is exists")
    return path



device="/gpu:0"
checkpoint_path= mkdir("./checkpoints/model.ckp")
checkpoint_iter_path=mkdir(os.path.join(checkpoint_path,".iter"))
final_model_path=mkdir("./checkpoints/model")

train_logdir=mkdir("./train_log/{}".format(TM))
test_logdir=mkdir("./test_log/{}".format(TM))

#params
lr=0.0001 #learning rate
epoch_n=
batch_size=
batch_n=
show_iter=100
checkpoint_iter=10000
classes= # the labels, classes=y_train or y_test

lambd=0.01# 正则系数
