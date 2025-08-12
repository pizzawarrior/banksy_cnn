import os
import json


def save_model_with_metadata(model,
                             history,
                             hyperparams,
                             save_dir,
                             metrics=None,
                             fold=None):
    '''
    save model with comprehensive metadata and training history.
    works for both cv models trained on each fold, as well as fully
    trained models.
    '''

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if fold is not None:
        # cross-validate models
        filename = f'cnn_{hyperparams["tile_h"]}x{hyperparams["tile_w"]}_overlap{hyperparams["overlap"]:.1f}_entropy{hyperparams["entropy_threshold"]:.1f}_{hyperparams["architecture"]}_fold{fold}'

    else:
        # fully trained models
        filename = f'cnn_{hyperparams["tile_h"]}x{hyperparams["tile_w"]}_overlap{hyperparams["overlap"]:.1f}_entropy{hyperparams["entropy_threshold"]:.1f}'

    model_path = os.path.join(save_dir, f'{filename}.keras')
    model.save(model_path)

    history_path = os.path.join(save_dir, f'{filename}_history.json')
    history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}

    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    # this works for both cv and fully trained models
    metadata = {
        'hyperparameters': hyperparams,
        'model_path': model_path,
        'history_path': history_path
    }

    # add these for cv models that have validation metrics and a fold to save
    if metrics is not None:
        metadata['metrics'] = metrics
        metadata['fold'] = fold

        print(f'Tile validation accuracy: {metrics["tile_accuracy"]:.4f}')
        print(f'Tile validation F1: {metrics["tile_f1"]:.4f}')
        print(f'Image validation accuracy: {metrics["img_accuracy"]:.4f}')
        print(f'Image validation F1: {metrics["img_f1"]:.4f}')

    metadata_path = os.path.join(save_dir, f'{filename}_metadata.json')

    # NOTE: this overwrites file contents, if file already exists
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'Model saved: {model_path}')

    return model_path, history_path, metadata_path
