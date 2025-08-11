import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from src.data_loader import get_images
from src.split_data import split_data_train_test
from src.make_dataset import create_tiles_dataset
from utils.model_builders import model_builders
from src.save_model import save_model_with_metadata
from src.evaluate import get_metrics




def train_full(X_train,
                y_train,
                hyperparams,
                save_dir='models/saved/fully_trained'):
    '''
    search thru all metadata files and find the best performing models.
    load the hyperparams of the top 5 UNIQUE model configurations.
    (prevent multiple folds from the same model config from counting as different models).
    train those 5 models and save their metrics.
    TODO: double check the training/ validation metrics that we have access to and can save.
    '''
    print(f'\n{"="*50}')
    print('Training with hyperparameters:')
    for key, value in hyperparams.items():
        print(f'{key}: {value}')
    print(f'{"="*50}')

    X_tiles, y_tiles, tiles_to_image = create_tiles_dataset(
    X_train, y_train,
    hyperparams['tile_h'], hyperparams['tile_w'],
    hyperparams['overlap'], hyperparams['entropy_threshold'],
    augment=True
    )

    print(f'Total training tiles: {X_tiles.shape[0]}')

    model = model_builders(hyperparams)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
    )

    epochs = 50

    history = model.fit(
        X_tiles, y_tiles,
        validation_split=0.1,  # holdout 10% for validation
        epochs=epochs,
        batch_size=hyperparams['batch_size'],
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Now we decide what metrics we want to display: tile-level and img-level
    # Then save the metrics, history, and metadata.
