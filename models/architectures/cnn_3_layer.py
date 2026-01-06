from tensorflow import keras
from tensorflow.keras import layers


# def cnn_3_layer(tile_h, tile_w, learning_rate=0.001):
#     '''
#     version 1: adds a convolutional layer, then relu activation, then batch normalization
#     version 2 below switches this to: conv layer, BN, then relu
#     '''
#     model = keras.Sequential()
#     model.add(layers.Input(shape=(tile_h, tile_w, 1)))
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Dropout(0.25))

#     model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Dropout(0.25))

#     model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Dropout(0.25))  # initially set to .5

#     model.add(layers.GlobalAveragePooling2D())

#     # classification
#     model.add(layers.Dense(128, activation='relu'))
#     model.add(layers.Dropout(0.5))  # binary classification: >0.5 = real, <0.5 = fake
#     model.add(layers.Dense(1, activation='sigmoid'))

#     # compile
#     optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#     model.compile(
#         optimizer=optimizer,
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )

#     return model


def cnn_3_layer(tile_h, tile_w, learning_rate=0.001):
    '''
    version 2: conv layer, BN, then relu
    this architecture references Casalegno, et al., 'Caries Detection w/...', 2019.
    '''
    model = keras.Sequential()
    model.add(layers.Input(shape=(tile_h, tile_w, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation=None, padding='same'))  # Experiment
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))  # Experiment
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(32, (3, 3), activation=None, padding='same'))  # Experiment
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))  # Experiment
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(32, (3, 3), activation=None, padding='same'))  # Experiment
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))  # Experiment
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))  # initially set to .5

    model.add(layers.GlobalAveragePooling2D())

    # classification
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # binary classification: >0.5 = real, <0.5 = fake
    model.add(layers.Dense(1, activation='sigmoid'))

    # compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
