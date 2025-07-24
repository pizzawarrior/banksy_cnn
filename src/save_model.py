import os
import json


def save_model_with_metadata(model, history, hyperparams, fold, metrics, save_dir='models/saved'):
    '''
    save model with comprehensive metadata and training history.
    '''

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f'cnn_{hyperparams["tile_h"]}x{hyperparams["tile_w"]}_overlap{hyperparams["overlap"]:.1f}_entropy{hyperparams["entropy_threshold"]:.1f}_{hyperparams["architecture"]}_fold{fold}'
    model_path = os.path.join(save_dir, f'{filename}.keras')
    model.save(model_path)

    history_path = os.path.join(save_dir, f'{filename}_history.json')  # save training history
    history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    metadata = {  # save metadata and metrics
        'hyperparameters': hyperparams,
        'fold': fold,
        'metrics': {k: float(v) for k, v in metrics.items()},
        'model_path': model_path,
        'history_path': history_path
    }

    metadata_path = os.path.join(save_dir, f'{filename}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'Model saved: {model_path}')
    print(f'Validation accuracy: {metrics["val_accuracy"]:.4f}')
    print(f'Validation F1: {metrics["val_f1"]:.4f}')

    return model_path, history_path, metadata_path
