from tensorflow import keras
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from src import converter_audio

_, X_test, _, y_test, _ = converter_audio.prepared_data_and_get_models()

trained_model = keras.models.load_model('./best_model.h5')
predictions = trained_model.predict_classes(X_test)

print(classification_report(y_test, to_categorical(predictions)))