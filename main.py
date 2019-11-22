from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import os
import numpy as np
import platform
# load data
from tflearn.datasets import cifar10
# (X, Y), (X_test, Y_test) = cifar10.load_data()
data_path = "./cifar-10-batches-py/cifar-10-batches-py"



# 读取文件
def load_pickle(f):
    version = platform.python_version_tuple() # 取python版本号
    if version[0] == '2':
        return  pickle.load(f) # pickle.load, 反序列化为python的数据类型
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)  # dict类型
        X = datadict['data']  # X, ndarray, 像素值
        Y = datadict['labels']  # Y, list, 标签, 分类

        # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
        # transpose，转置
        # astype，复制，同时指定类型
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []  # list
    ys = []

    # 训练集batch 1～5
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)  # 在list尾部添加对象X, x = [..., [X]]
        ys.append(Y)
    Xtr = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
    Ytr = np.concatenate(ys)
    del X, Y

    # 测试集
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
X, Y, X_test, Y_test = load_CIFAR10(data_path)
X, Y = shuffle(X, Y)
# transform to 10 one-hot code
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)
# Real time data preprocessing
# std and 0 center
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
# real time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
# Cov
network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation="relu")
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation="relu")
network = conv_2d(network, 64, 3, activation="relu")
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation="relu")
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation="softmax")
network = regression(network, optimizer="adam", loss='categorical_crossentropy', learning_rate=0.001)

# train
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir="./log")
model.fit(X, Y, n_epoch=1, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=96, run_id="cifar10_cnn")

