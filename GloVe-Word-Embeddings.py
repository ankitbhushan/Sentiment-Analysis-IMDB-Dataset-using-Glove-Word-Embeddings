import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Processing the labels of the raw IMDB data
PROJECT_ROOT = os.getcwd()
main_dir_path = os.path.join(PROJECT_ROOT, 'imdb')
train_dir_path = os.path.join(main_dir_path, 'train')

texts = []
labels = []
for label_type in ['pos', 'neg']:
    path = os.path.join(train_dir_path, label_type)
    for fn in os.listdir(path):
        if fn[-4:] == '.txt':
            f = open(os.path.join(path, fn), encoding='utf8')
            texts.append(f.read())
            f.close()
            if label_type == 'pos':
                labels.append(1)
            else:
                labels.append(0)

# print(texts[:10])
# print(labels[:10])

# Tokenizing the text of the raw IMDB data
vocab_size = 10000
training_sample = 200
validation_samples = 10000
max_len = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<oov>')
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=max_len)

labels = np.array(labels)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_sample]
y_train = labels[:training_sample]
x_val = data[training_sample: training_sample + validation_samples]
y_val = labels[training_sample: training_sample + validation_samples]

# PREPROCESSING THE EMBEDDINGS - GloVe
glove_path = os.path.join(PROJECT_ROOT, 'glove.6B')
f = open(os.path.join(glove_path, 'glove.6B.100d.txt'), encoding='utf8')
embeddings_index = {}
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index[word] = np.array(values[1:], dtype='float32')
f.close()

print(f'Found {len(embeddings_index)} word vectors.')

# Preparing the GloVe word-embeddings matrix
embedding_dim = 100
embeddings_matrix = np.zeros((vocab_size, embedding_dim))
for word, value in word_index.items():
    if value < vocab_size:
        word_vector = embeddings_index.get(word)
        if word_vector is not None:
            embeddings_matrix[value] = word_vector

# DEFINING A MODEL
model = models.Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# Loading pretrained word embeddings into the Embedding layer
model.layers[0].set_weights([embeddings_matrix])
model.layers[0].trainable = False

# Training and evaluation
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

# Plotting the results
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(len(train_acc))

plt.plot(epochs, train_acc, 'bo', label='Training_acc')
plt.plot(epochs, val_acc, 'b', label='validation_acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, train_loss, 'bo', label='Training_loss')
plt.plot(epochs, val_loss, 'b', label='validation_loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
