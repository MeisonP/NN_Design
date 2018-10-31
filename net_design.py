#coding:utf-8
# 2018.10.30
# mason_P first nn design for classify
from config import *
#from dataset import *
import tensorflow as tf



##class net
class network:
    def __init__(self):
        pass

    #dense_layer
    @classmethod
    def dense_layer(cls,input_tensor,hidden_number,name_):
        regularization = tf.contrib.layers.l2_regularizer(lambd) #对fc网络使用正则化
        fc = tf.layers.dense(input_tensor,hidden_number,activation=tf.nn.relu,
                            kernel_regularizer=regularization,name=name_)
        return fc

    #net-build
    @classmethod
    @func_track
    def net_build(cls,input_,output_dim):
        net = {}
        net_input=input_
        net["fc1"]=cls.dense_layer(net_input,6,"fc1")
        net["fc2]"]=cls.dense_layer(net["fc1"],4,"fc2")
        net["output"]=cls.dense_layer(net["fc2]"],output_dim,"output")
        #the number of neural in output layer is the length of labels/ classes

        return net





