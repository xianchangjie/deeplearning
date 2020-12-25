import tensorflow as tf
import os

train_dir_path = '/Users/changjiexian/Documents/tc/ds/aclImdb'
train_dir = os.path.join(train_dir_path, 'train')

labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        f = open(os.path.join(dir_name, fname))
        texts.append(f.read())
        f.close()
        if label_type == 'neg':
            labels.append(0)
        else:
            labels.append(1)

print(labels[0])
print(texts[0])

#在100个单词后截断评论
maxlen = 100
#在200个样本上进行训练
training_samples = 200
#在10000个样本上验证
validation_samples = 10000
#只考虑数据集前10000个最常见的单词
max_words = 10000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)



