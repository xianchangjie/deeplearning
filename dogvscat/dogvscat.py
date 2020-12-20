from tensorflow.keras.datasets import imdb
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_dir_path = '/Users/changjiexian/Documents/tc/ds/dogvscat/train'
val_dir_path = '/Users/changjiexian/Documents/tc/ds/dogvscat/val'
test_dir_path = '/Users/changjiexian/Documents/tc/ds/dogvscat/test'

# print(len(os.listdir(train_dir_path)))
# print(len(os.listdir(val_dir_path)))
# print(len(os.listdir(test_dir_path)))

train_data_gen = ImageDataGenerator(rescale=1. / 255)
val_data_gen = ImageDataGenerator(rescale=1. / 255)
test_data_gen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_data_gen.flow_from_directory(train_dir_path, target_size=(150, 150), batch_size=20,
                                               class_mode='binary')
val_gen = val_data_gen.flow_from_directory(val_dir_path, target_size=(150, 150), batch_size=20,
                                           class_mode='binary')

# for data_batch, labels_batch in train_gen:
#     print(data_batch)
#     print(labels_batch)
#     break

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

history = model.fit(train_gen, steps_per_epoch=100, epochs=30, validation_data=val_gen, validation_steps=50)
model.save('cats_and_dogs_small_2.h5')


# print(history.history.keys())

acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Acc')
plt.plot(epochs, val_acc, 'b', label='Val Acc')
plt.title('Train & Val acc')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
