# -*- coding: utf-8 -*-
__author__ = 'sn'


import tensorflow as tf
import numpy as np
#添加神经网络层
def add_layer(inputs,in_size,out_size,activation_function=None):
    #构建权重
    weights=tf.Variable(tf.random_normal([in_size,out_size]))
    #构建偏置
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    #矩阵相乘
    Wx_plus_b=tf.matmul(inputs,weights)+biases
    if activation_function is None:
        out_puts=Wx_plus_b
    else:
        out_puts=activation_function(Wx_plus_b)
    return out_puts#得到输出数据
if __name__=='__main__':
    #定义占位符输入变量
    xs=tf.placeholder(tf.float32,[None,1])
    ys=tf.placeholder(tf.float32,[None,1])
    #构建隐藏层
    h1=add_layer(xs,1,20,activation_function=tf.nn.relu)
    #构建输出层
    prediction = add_layer(h1,20,1,activation_function=None)
    #计算预测值和真实值之间的误差
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    #构造数据
    x_data=np.linspace(-1,1,300)[:,np.newaxis]
    noise=np.random.normal(0,0.05,x_data.shape)
    y_data=np.square(x_data)-0.5+noise

    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)#初始化变量
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%50 == 0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))




