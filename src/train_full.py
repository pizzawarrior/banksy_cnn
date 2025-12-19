from tensorflow import keras
from src.make_dataset import create_tiles_dataset
from src.utils import model_builders
from src.save_model_with_metadata import save_model_with_metadata


def train_full(X_train,
               y_train,
               hyperparams,
               save_dir='models/saved/fully_trained'):
    '''
    - train a model on the full dataset using the hyperparams of a saved cv-model
    accepts:
        - training image set: np.array, X_train,
        - corresponding training lables: np.array, y_train
        - hyperparams: dict, contains saved model params
    - save the model and performance metrics
    TODO: double check the training/ validation metrics that we have access to and can save.
    '''
    print(f'\n{"="*50}')
    print('Training with hyperparameters:')
    for k, v in hyperparams.items():
        print(f'{k}: {v}')
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
    batch_size = hyperparams.get('batch_size', 32)  # set default to 32

    history = model.fit(
        X_tiles, y_tiles,
        validation_split=0.1,  # holdout 10% for validation
        epochs=epochs,
        batch_size=batch_size,
        # class_weight={0: 9.0},  # experiment with weighting 0 class higher
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    result_paths = save_model_with_metadata(model, history, hyperparams, save_dir)

    return model, history, result_paths
