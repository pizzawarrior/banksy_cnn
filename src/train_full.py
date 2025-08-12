from tensorflow import keras
from src.make_dataset import create_tiles_dataset
from src.utils import model_builders
from src.save_model import save_model_with_metadata


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

    X_tiles, y_tiles, _ = create_tiles_dataset(
        X_train, y_train,
        hyperparams['tile_h'], hyperparams['tile_w'],
        hyperparams['overlap'], hyperparams['entropy_threshold'],
        augment=True
    )

    print(f'Total training tiles: {X_tiles.shape[0]}')

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
    )

    model, early_stopping, reduce_lr = model_builders(hyperparams)
    epochs = 50
    batch_size = hyperparams.get('batch_size', 64)

    history = model.fit(
        X_tiles, y_tiles,
        validation_split=0.1,  # holdout 10% for validation
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    result_paths = save_model_with_metadata(model, history, hyperparams, save_dir)

    return model, history, result_paths
