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
                        epochs=60,
                        save_dir='models/saved'):
    '''
    train model using k-fold cross-validation with integrated saving.
    hyperparams are a dict with:
        - tile_h, tile_w, overlap, entropy_threshold, architecture, learning_rate
    '''

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

        # create training tile datasets
        X_train_tiles, y_train_tiles, _ = create_tiles_dataset(
            train_images, train_labels,
            hyperparams['tile_h'], hyperparams['tile_w'],
            hyperparams['overlap'], hyperparams['entropy_threshold'],
            augment=True
        )
        # create validation tile datasets
        X_val_tiles, y_val_tiles, val_tile_to_image = create_tiles_dataset(
            val_images, val_labels,
            hyperparams['tile_h'], hyperparams['tile_w'],
            hyperparams['overlap'], hyperparams['entropy_threshold'],
            augment=False
        )

        print(f'Training tiles: {X_train_tiles.shape[0]}, Validation tiles: {X_val_tiles.shape[0]}')

        if hyperparams['architecture'] == '3layer':
            model = cnn_3_layer(
                hyperparams['tile_h'], hyperparams['tile_w'], hyperparams['learning_rate']
            )
        elif hyperparams['architecture'] == '5layer':
            model = cnn_5_layer(
                hyperparams['tile_h'], hyperparams['tile_w'], hyperparams['learning_rate']
            )
        else:
            raise ValueError(f'Model name {hyperparams["architecture"]} is invalid. \
                Must be either `3layer` or `5layer`.')

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
        )

        history = model.fit(
            X_train_tiles, y_train_tiles,
            validation_data=(X_val_tiles, y_val_tiles),
            epochs=epochs,
            batch_size=hyperparams['batch_size'],
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        tile_val_pred_proba = model.predict(X_val_tiles)
        classification_threshold = hyperparams['classification_threshold']
        # validation classification of each tile NOT image
        tile_val_pred = (tile_val_pred_proba > classification_threshold).astype(int).flatten()

        # track tiles and labels to each image for image level classification
        # during validation
        img_preds = {}
        img_true_labels = {}

        for tile_idx, img_idx in enumerate(val_tile_to_image):
            if img_idx not in img_preds:
                img_preds[img_idx] = []
                img_true_labels[img_idx] = val_labels[img_idx]
            img_preds[img_idx].append(tile_val_pred_proba[tile_idx][0])

        img_level_preds = []
        img_level_true_labels = []

        for img_idx in sorted(img_preds.keys()):
            avg_pred = np.mean(img_preds[img_idx])
            img_level_preds.append(avg_pred)
            img_level_true_labels.append(img_true_labels[img_idx])

        img_pred_binary = (np.array(img_level_preds) > classification_threshold).astype(int)

        img_val_metrics = calc_metrics(img_level_true_labels, img_pred_binary, img_level_preds, 'img')
        tile_val_metrics = calc_metrics(y_val_tiles, tile_val_pred, tile_val_pred_proba.flatten(), 'tile')
        val_metrics = tile_val_metrics | img_val_metrics  # combine dicts

        model_paths = save_model_with_metadata(
            model, history, hyperparams, fold, val_metrics, save_dir
        )

        fold_results.append(val_metrics)
        saved_models.append(model_paths)

    # display cv summary once validation is done, for tiles and imgs
    cv_summary = {}
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
        tile_values = [fold[f'tile_{metric}'] for fold in fold_results]
        cv_summary[f'tile_{metric}_mean'] = np.mean(tile_values)
        img_values = [fold[f'img_{metric}'] for fold in fold_results]
        cv_summary[f'img_{metric}_mean'] = np.mean(img_values)

    print(f'\n{"="*50}')
    print('Cross-Validation Results:')

    for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
        tile_mean_val = cv_summary[f'tile_{metric}_mean']
        print(f'{metric}: {tile_mean_val:.4f}')
        img_mean_val = cv_summary[f'img_{metric}_mean']
        print(f'{metric}: {img_mean_val:.4f}')
    print(f'{"="*50}')

    return cv_summary, fold_results, saved_models
