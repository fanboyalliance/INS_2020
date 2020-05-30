from src import converter_audio
from src.email_sender import sent_email_notification
from tensorflow.keras import callbacks
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

# set gpu visible to tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

model_number = 1
epochs = 50


class EmailNotificationCallback(callbacks.Callback):
    def __init__(self, logs=None):
        self.time_now = 0
        self.epoch_counter = 0
        self.epoch_accuracy = 0
        self.best_epoch_accuracy = 0
        self.best_epoch = 0

    def on_train_begin(self, logs=None):
        self.time_now = datetime.datetime.now()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_counter += 1

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_accuracy = logs['val_accuracy']
        if self.best_epoch_accuracy == 0:
            self.best_epoch_accuracy = self.epoch_accuracy

        if self.epoch_accuracy > self.best_epoch_accuracy:
            self.best_epoch_accuracy = self.epoch_accuracy
            self.best_epoch = self.epoch_counter

    # after model training set best accuracy with epoch to email
    def on_train_end(self, logs=None):
        try:
            to = 'alex.raswqa@gmail.com'
            val_accuracy = round(self.best_epoch_accuracy * 100, 2)
            time_to_train = datetime.datetime.now() - self.time_now
            message = 'Model #' + str(model_number) + ' has just finished \nIt took ' + str(
                time_to_train) + ' minutes' + '\nBest validation accuracy equals ' + str(val_accuracy) + '% on ' + str(
                self.best_epoch) + ' epoch'
            print(message)
            sent_email_notification(to, message)
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)


# gpu limit
def set_gpu_limit_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def create_plot(history_dict):
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    plt.plot(epochs, loss_values, 'r', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss" + str(model_number) + ".png")
    plt.clf()

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("accuracy" + str(model_number) + ".png")


def get_callbacks():
    # tensorboard view
    tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=1,
                                        write_graph=True, write_images=True, profile_batch=100000000)

    # save best model by accuracy
    checkpoint = callbacks.ModelCheckpoint('best_model.h5', verbose=1, monitor='val_accuracy',
                                           save_best_only=True, mode='auto')

    return [tensorboard, EmailNotificationCallback(), checkpoint]


def test_this_model(model, x_test, y_test):
    predictions = model.predict_classes(x_test)
    print(classification_report(y_test, to_categorical(predictions)))


try:
    set_gpu_limit_memory()

    # use gpu
    with tf.device('/gpu:0'):
        train_callbacks = get_callbacks()

        X_train, X_test, y_train, y_test, models = converter_audio.prepared_data_and_get_models()

        for model in models:
            try:
                print('try ' + str(model_number))
                history = model.fit(X_train, y_train, batch_size=64, epochs=epochs, verbose=1, validation_split=0.1,
                                    callbacks=train_callbacks)
                create_plot(history.history)
                test_this_model(model, X_test, y_test)
                model_number += 1
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
except RuntimeError as e:
    print(e)
