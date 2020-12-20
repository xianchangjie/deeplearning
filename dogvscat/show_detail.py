from tensorflow.keras.datasets import imdb
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img = image.load_img('/Users/changjiexian/Documents/tc/ds/dogvscat/test/186.jpg',
                     target_size=(150, 150))
img_t = image.img_to_array(img)
img_array = np.expand_dims(img_t, axis=0)
img_array /= 255.
#
# plt.imshow(img_array)
# plt.show()

model = tf.keras.models.load_model('cats_and_dogs_small_1.h5')

res = model.predict(img_array)
print(res)
# layer_outputs = [layer.output for layer in model.layers[:8]]

# activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(img_array)
# first_layer_activation = activations[0]
# plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
# plt.show()

# layer_names = []
# for layer in model.layers[:8]:
#     layer_names.append(layer.name)

# images_per_row = 16
#
# for layer_name, layer_activation in zip(layer_names, activations):
#     n_features = layer_activation.shape[-1]
#     # print(layer_name)
#     # print(layer_activation.shape)
#     # print(layer_activation.shape[-1])
#     size = layer_activation.shape[1]
#     n_cols = n_features
#     # conv2d
#     # (1, 148, 148, 32)
#     # 32
#     # max_pooling2d
#     # (1, 74, 74, 32)
#     # (148*32,16* 148)
#     display_grid = np.zeros((size * n_cols, images_per_row * size))
#     for col in range(n_cols):
#         for row in range(images_per_row):
#             channel_image = layer_activation[0, :, :, col * images_per_row + row]
#             channel_image -= channel_image.mean()
#             channel_image /= channel_image.std()
#             channel_image *= 64
#             channel_image += 128
#             channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#             display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
#             scale = 1. / size
#             plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
#             plt.title(layer_name)
#             plt.grid(False)
#             plt.imshow(display_grid, aspect='auto', cmap='viridis')
#             plt.show()
