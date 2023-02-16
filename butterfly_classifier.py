import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Rescaling, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class ButterflyClassifier():
    
    scale = 1./127.5
    offset = -1

    def __init__(self, image_size: tuple, batch_size: int, src_training: str, src_validation: str, src_test: str):
        self.image_size = image_size
        self.batch_size = batch_size
        self.__process_datasets(src_training, src_validation, src_test)
        self.__create_model()

    def __process_datasets(self, src_training: str, src_validation: str, src_test: str):
        self.train_dataset = keras.preprocessing.image_dataset_from_directory(
            directory=src_training,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode='categorical'
        )

        self.valid_dataset = keras.preprocessing.image_dataset_from_directory(
            directory=src_validation,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode='categorical'
        )

        self.test_dataset = keras.preprocessing.image_dataset_from_directory(
            directory=src_test,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode='categorical'
        )

        self.train_dataset = self.train_dataset.prefetch(buffer_size=32)
        self.valid_dataset = self.valid_dataset.prefetch(buffer_size=32)
        self.test_dataset = self.test_dataset.prefetch(buffer_size=32)

    def __create_model(self):
        self.model = keras.Sequential()
        input_shape = self.image_size + (3,)

        self.model.add(Rescaling(
            scale=self.scale,
            offset=-self.offset,
            input_shape=input_shape
        ))

        self.model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(6, activation='softmax'))

        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adam(1e-3),
                    metrics=['accuracy'])

    def train(self, epochs: int):
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, restore_best_weights=True)

        h = self.model.fit(
            self.train_dataset,
            epochs=epochs, 
            validation_data=self.valid_dataset,
            callbacks = [es]
        )

        return h

    def valid_confussion_matrix(self):
        results = np.concatenate([(y, self.model.predict(x=x)) for x, y in self.valid_dataset], axis=1)

        predictions = np.argmax(results[0], axis=1)
        labels = np.argmax(results[1], axis=1)

        cf_matrix = confusion_matrix(labels, predictions)
        sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues")

        print(classification_report(labels, predictions, digits = 4))

    def test_confussion_matrix(self):
        results = np.concatenate([(y, self.model.predict(x=x)) for x, y in self.test_dataset], axis=1)

        predictions = np.argmax(results[0], axis=1)
        labels = np.argmax(results[1], axis=1)

        cf_matrix = confusion_matrix(labels, predictions)
        sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues")

        print(classification_report(labels, predictions, digits = 4))