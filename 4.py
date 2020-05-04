import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from keras.models import Sequential
import numpy as np
from PIL import Image
from keras import optimizers

mnist = tf.keras.datasets.mnist

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def build_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

def get_image(filename):
    im = Image.open(filename)
    im = im.resize((28, 28))
    im = np.dot(np.asarray(im), np.array([1/3, 1/3, 1/3]))
    im /= 255
    im = 1 - im
    im = im.reshape((1, 28, 28))

    return im

model = build_model()
model.compile(optimizer=optimizers.Adam(learning_rate=0.1),loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))

plt.title('Training and test accuracy Adam 0.1')
plt.plot(history.history['accuracy'], 'r', label='Training acc')
plt.plot(history.history['val_accuracy'], 'b', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.clf()

plt.plot(history.history['loss'], 'r', label='Training loss')
plt.plot(history.history['val_loss'], 'b', label='Validation loss')
plt.title('Training and validation accuracy Adam 0.1')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.clf()




