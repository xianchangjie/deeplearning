# -*- coding: UTF-8 -*-
from sklearn import datasets
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# 导入数据集
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 打乱数据集
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 分割训练数据/测试数据
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 配对输入特征->标签
train__batch = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(30)
test__batch = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(30)

# 定义可训练参数
w = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1))
b = tf.Variable(tf.random.truncated_normal([3, ], stddev=0.1))

lr = 0.1  # 学习率为0.1
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 500  # 循环500轮
loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和

# print(y_train)
# print(len(y_train))
# print(tf.one_hot(y_train,3))

# 数据集级别的循环，每个epoch循环一次数据集
for epoch in range(epoch):
    # batch级别的循环 ，每个step循环一个batch
    for step, (x_train, y_train) in enumerate(train__batch):
        # with结构记录梯度信息
        with tf.GradientTape() as tape:
            # 神经网络乘加运算
            # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
            y = tf.nn.softmax(tf.matmul(x_train, w) + b)
            # 将标签值转换为独热码格式，方便计算loss和accuracy
            y_ = tf.one_hot(y_train, 3)
            # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss = tf.reduce_mean(tf.square(y_ - y))
            # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
            loss_all += loss.numpy()
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w, b])
        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        # 参数w自更新
        w.assign_sub(lr * grads[0])
        # 参数b自更新
        b.assign_sub(lr * grads[1])
    # 每个epoch，打印loss信息
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备

    # 测试部分
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test__batch:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w) + b
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")