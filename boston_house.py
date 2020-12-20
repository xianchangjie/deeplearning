from tensorflow.keras.datasets import imdb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
print(len(x_train))
print(len(y_train))
print(x_train[0])
print(y_train[0])