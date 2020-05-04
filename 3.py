import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model

k = 4
num_val_samples = len(train_data) // k
num_epochs = 50
all_scores = []
val_mae_histories = []
mae_histories = []
epochs = range(1, num_epochs + 1)

for i in range(k):
    print('processing fold #', i)

    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()

    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(val_data, val_targets))
    history_dict = history.history

    mae_hist = history_dict['mae']
    mae_histories.append(mae_hist)

    val_mae_hist = history_dict['val_mae']
    val_mae_histories.append(val_mae_hist)

    plt.plot(epochs, mae_hist, 'r', label='Training mean absolute error')
    plt.plot(epochs, val_mae_hist, 'b', label='Validation mean absolute error')
    plt.title('Training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('Mae')
    plt.legend()
    plt.show()

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)

    all_scores.append(val_mae)

print(np.mean(all_scores))
plt.clf()
plt.plot(epochs, np.mean(mae_histories, axis=0), 'r', label='Training mae')
plt.plot(epochs, np.mean(val_mae_histories, axis=0), 'b', label='Validation mae')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()