import tensorflow as tf
import numpy as np




# a = tf.constant([1,5],tf.int64)
# print(a)
# a = np.arange(0,5)
# print(a)
# b=tf.convert_to_tensor(a,tf.int64)
# print(b)

# print(tf.zeros([1,3]))
# print(tf.ones(4))
# print(tf.fill([2,5],8))
# print(tf.random.normal([2,2],0,1))
# print(tf.random.truncated_normal([2,2],0,1))
# print(tf.random.uniform([4,1],0,1))

# features = tf.constant([12, 23, 11, 17])
# labels = tf.constant([0, 1, 1, 0])
# ds = tf.data.Dataset.from_tensor_slices((features, labels))
# print(ds)
# for ele in ds:
#     print(ele)

# with tf.GradientTape() as tape:
#     w = tf.Variable(tf.constant(3.0))
#     loss = tf.pow(w,2)
#
# print(tf.one_hot(tf.constant([0,1,2]),2))

# print(tf.random.truncated_normal(3))
# print(tf.random.truncated_normal([3]))
# print(tf.random.truncated_normal([3,]))


# print(np.mgrid[1:3:1,7:9:1])