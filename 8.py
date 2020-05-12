import numpy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential

filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

seq_length = 100

dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

def test_network(epoch):
    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(list(set(raw_text)))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]

    resultText = []
    # generate characters
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        resultText.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    save_text = 'Epoch ' + str(epoch) + '\n' + ''.join(resultText) + '\n'
    f = open("result.txt", "a")
    f.write(save_text)
    f.close()

class CustomCallback(Callback):
    def __init__(self):
        self.__epochCounter = 0

    def get_x(self):
        return self.__epochCounter

    def set_x(self, x):
        self.__epochCounter = x

    epochCounter = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochCounter = self.epochCounter + 1
        if self.epochCounter == 1 or self.epochCounter % 3 == 0:
            print("End epoch {} of training".format(epoch))
            test_network(self.epochCounter)

tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True,
                         profile_batch=100000000)
callbacks_list = [
    checkpoint,
    tbCallBack,
    CustomCallback()
]
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)