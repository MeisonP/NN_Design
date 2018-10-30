# conding: utf-8
# 2018.10.30
# mason_P first nn design for classify
from config import *
import tensorflow as tf
import numpy as np
import os





##class data
class data:
    @classmethod
    @func_track
    def load_images(self):
        return img

    @classmethod
    @func_track
    def pre_process(self):
        return img





##class net
class net:
    #dense_layer
    @classmethod
    @func_track
    def dense_layer(self,input_tensor,hedden_number,name_):
        regularizer = tf.contrib.layers.l2_regularizer(lambd) # using zheng-ze-hua in fc
        fc = tf.layers.dense(input_tensor,hedden_number,activation=tf.nn.relu,
                            kernel_regularizer=regularizer,name=name_)
        return fc

    #net-build
    @classmethod
    @func_track
    def net_build(self,input):
        net = {}
        net_input=input
        net["fc1"]=self.dense_layer(net_input,6,"fc1")
        net["fc2]"]=self.dense_layer(net["fc1"],4,"fc2")
        net["output"]=self.dense_layer(net["fc2]"],classes,"output")
        #the number of neural in output layer is the length of labels/ classes

        return net





