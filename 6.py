import numpy as np
from keras import Sequential
from keras import layers

from keras.datasets import imdb

dimension = 10000

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dimension)

data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

def vectorize(sequences, dim=dimension):
    results = np.zeros((len(sequences), dim))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1

    return results

data = vectorize(data)
targets = np.array(targets).astype("float32")

test_x = data[:10000]
test_y = targets[:10000]

train_x = data[10000:]
train_y = targets[10000:]

def build_model():
    model = Sequential()
    # Input - Layer
    model.add(layers.Dense(50, activation="relu", input_shape=(dimension,)))
    # Hidden - Layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    # Output- Layer

    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    return model

def input_user_review():
    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))

    text = input('Enter user text')
    reviews = [text]
    word_index = imdb.get_word_index()
    number_reviews = []
    for review in reviews:
        single_number_review = []
        for w in review:
            if w in word_index and word_index[w] < dimension:
                single_number_review.append(word_index[w])
        number_reviews.append(single_number_review)
    for i in range(len(number_reviews)):
        vectorized_review = vectorize([number_reviews[i]])
        accuracy = model.predict(vectorized_review)[0][0]
        print('Review "' + reviews[i] + '"\n' + ' has a ' + str(accuracy) + ' accuracy to be positive')
input_user_review()
