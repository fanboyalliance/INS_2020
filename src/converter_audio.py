import numpy as np
import librosa
from librosa import display
import os
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from src import converter_audio, model_creator
from sklearn.model_selection import train_test_split

def draw_spectrogram(m_slaney, label):
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    display.specshow(m_slaney, x_axis='time')
    plt.colorbar()
    plt.title('Number ' + label)
    plt.tight_layout()
    plt.savefig(label + ".png")
    plt.show()

def wav2mfcc(file_path, label, max_pad_len=20):
    wave, _ = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    # Compute MFCC features from the raw signal
    # sr = 8000 ~ 8 kHz as default for this dataset
    mfcc = librosa.feature.mfcc(wave, sr=8000)

    # draw spectrogram
    #     showPictures(mfcc, label)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mfcc

# get all files from "recordings" folder
def get_sound_tensors_with_labels():
    labels = []
    mfccs = []

    for f in os.listdir('./recordings'):
        if f.endswith('.wav'):
            # parse label
            label = f.split('_')[0]

            mfccs.append(wav2mfcc('./recordings/' + f, label))

            labels.append(label)

    return np.asarray(mfccs), to_categorical(labels)

# get prepared data and 8 models to train
def prepared_data_and_get_models():
    mfccs, labels = get_sound_tensors_with_labels()
    dim_1 = mfccs.shape[1]
    dim_2 = mfccs.shape[2]
    channels = 1

    mfccs_copy = mfccs
    X = mfccs_copy.reshape((mfccs.shape[0], dim_1, dim_2, channels))
    y = labels

    input_shape = (dim_1, dim_2, channels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    models = [
        model_creator.get_cnn_model_1(input_shape),
        model_creator.get_cnn_model_2(input_shape),
        model_creator.get_cnn_model_3(input_shape),
        model_creator.get_cnn_model_4(input_shape),
        model_creator.get_cnn_model_5(input_shape),
        model_creator.get_cnn_model_6(input_shape),
        model_creator.get_cnn_model_7(input_shape),
        model_creator.get_cnn_model_8(input_shape)
              ]

    return X_train, X_test, y_train, y_test, models
