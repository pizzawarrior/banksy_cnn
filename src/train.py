import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from src.make_dataset import create_tiles_dataset
from src.save_model import save_model_with_metadata
from src.evaluate import calc_metrics
from models.architectures.cnn_3_layer import cnn_3_layer
from models.architectures.cnn_5_layer import cnn_5_layer


def train_model_with_cv(X_train,
                        y_train,
                        hyperparams,
                        n_folds=4,
                        epochs=50,
                        batch_size=16,
                        save_dir='models/saved'):
    '''
    train model using k-fold cross-validation with integrated saving.
    hyperparams are a dict with:
        - tile_h, tile_w, overlap, entropy_threshold, architecture, learning_rate
    '''

    # print hyperparams
    print(f'\n{"="*50}')
    print('Training with hyperparameters:')
    for key, value in hyperparams.items():
        print(f'{key}: {value}')
    print(f'{"="*50}')

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=420)

    fold_results = []
    saved_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f'\n--- Fold {fold}/{n_folds} ---')

        train_images = [X_train[i] for i in train_idx]
        train_labels = [y_train[i] for i in train_idx]
        val_images = [X_train[i] for i in val_idx]
        val_labels = [y_train[i] for i in val_idx]

        X_train_tiles, y_train_tiles = create_tiles_dataset(  # create training tile datasets
            train_images, train_labels,
            hyperparams['tile_h'], hyperparams['tile_w'],
            hyperparams['overlap'], hyperparams['entropy_threshold'],
            augment=True
        )

        X_val_tiles, y_val_tiles = create_tiles_dataset(  # create validation tile datasets
            val_images, val_labels,
            hyperparams['tile_h'], hyperparams['tile_w'],
            hyperparams['overlap'], hyperparams['entropy_threshold'],
            augment=False
        )

        print(f'Training tiles: {X_train_tiles.shape[0]}, Validation tiles: {X_val_tiles.shape[0]}')

        # create and train model
        if hyperparams['architecture'] == '3layer':
            model = cnn_3_layer(
                hyperparams['tile_h'], hyperparams['tile_w'], hyperparams['learning_rate']
            )
        elif hyperparams['architecture'] == '5layer':
            model = cnn_5_layer(
                hyperparams['tile_h'], hyperparams['tile_w'], hyperparams['learning_rate']
            )
        else:
            return f'Model name {hyperparams["architecture"]} is invalid. \
                Must be either `3layer` or `5layer`.'

        early_stopping = keras.callbacks.EarlyStopping(  # callbacks
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
        )

        history = model.fit(
            X_train_tiles, y_train_tiles,
            validation_data=(X_val_tiles, y_val_tiles),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        val_pred_proba = model.predict(X_val_tiles)
        val_pred = (val_pred_proba > 0.5).astype(int).flatten()  # NOTE threshold, can be changed

        val_metrics = calc_metrics(y_val_tiles, val_pred, val_pred_proba.flatten())
        val_metrics['val_accuracy'] = val_metrics['accuracy']
        val_metrics['val_f1'] = val_metrics['f1']
        val_metrics['val_precision'] = val_metrics['precision']
        val_metrics['val_recall'] = val_metrics['recall']
        val_metrics['val_auc'] = val_metrics['auc']

        model_paths = save_model_with_metadata(  # save model with pipeline
            model, history, hyperparams, fold, val_metrics, save_dir
        )

        fold_results.append(val_metrics)
        saved_models.append(model_paths)

    cv_summary = {}  # calc cv summary
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
        values = [fold[metric] for fold in fold_results]
        cv_summary[f'{metric}_mean'] = np.mean(values)
        cv_summary[f'{metric}_std'] = np.std(values)

    print(f'\n{"="*50}')
    print('Cross-Validation Results:')

    for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
        mean_val = cv_summary[f'{metric}_mean']
        std_val = cv_summary[f'{metric}_std']
        print(f'{metric}: {mean_val:.4f} ± {std_val:.4f}')
    print(f'{"="*50}')

    return cv_summary, fold_results, saved_models
