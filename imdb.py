from tensorflow.keras.datasets import imdb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)

y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

x_val = x_test[:10000]
x_test = x_test[10000:]

y_val = y_test[:10000]
y_test = y_test[10000:]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.02),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.binary_accuracy])

history = model.fit(x_train, y_train, batch_size=512, epochs=20, validation_data=(x_val, y_val))
his_dic = history.history
# print(his_dic.keys())
# dict_keys(['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])
loss_values = his_dic['loss']
acc_values = his_dic['binary_accuracy']
val_loss_values = his_dic['val_loss']
val_acc_values = his_dic['val_binary_accuracy']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs,loss_values,'bo',label='training loss')
plt.plot(epochs,acc_values,'ro',label='training acc')
plt.plot(epochs,val_loss_values,'b',label='val loss')
plt.plot(epochs,val_acc_values,'r',label='val acc')
plt.title("Loss & Acc")
plt.xlabel("epochs")
plt.ylabel("loss & acc")
plt.legend
plt.show()
