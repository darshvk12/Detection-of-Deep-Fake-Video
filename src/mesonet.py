from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense

class Meso4:
    def __init__(self, learning_rate=0.001):
        self.model = Sequential()

        # Layer 1
        self.model.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(BatchNormalization())

        # Layer 2
        self.model.add(Conv2D(8, (5, 5), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(BatchNormalization())

        # Layer 3
        self.model.add(Conv2D(16, (5, 5), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(BatchNormalization())

        # Layer 4
        self.model.add(Conv2D(16, (5, 5), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def load(self, weights_path):
        self.model.load_weights(weights_path)

    def predict(self, x):
        return self.model.predict(x)

