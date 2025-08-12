import json
import os
from tensorflow import keras
from models.architectures.cnn_3_layer import cnn_3_layer
from models.architectures.cnn_5_layer import cnn_5_layer
from src.get_metadata import get_metadata
from src.data_loader import get_images
from src.split_data import split_data_train_test


def model_builders(hyperparams):
    model_options = {
        '3layer': cnn_3_layer,
        '5layer': cnn_5_layer,
    }

    try:
        builder = model_options[hyperparams['architecture']]
        model = builder(hyperparams['tile_h'], hyperparams['tile_w'], hyperparams['learning_rate'])
    except KeyError:
        raise ValueError(
            f"Model name '{hyperparams['architecture']}' is invalid. Must be one of: {', '.join(model_options.keys())}."
        )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
    )

    return model, early_stopping, reduce_lr


def get_metrics(dir_path='models/saved/cv_results'):
    '''
    TODO: add descriptions
    '''
    metadata_dict = {}

    # find all metadata paths, parse the files, and add the metrics to a dict
    for dir_path, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith('metadata.json'):
                file_path = os.path.join(dir_path, filename)
                metadata_dict[file_path] = get_metadata(file_path)
    if not metadata_dict:
        return f'There are no metadata files in the current directory: {dir_path}'
    return metadata_dict


def get_top_n_models(metadata_dict, n=5):
    '''
    TODO: add descriptions
    '''
    ranked_models = dict(sorted(
        metadata_dict.items(),
        key=lambda model: (
            model[1]['precision'],
            model[1]['f1'],
            model[1]['accuracy']),
        reverse=True
    ))

    unique_models = {}
    seen_configs = set()
    i = 0
    print(f'The top {n} models, in order, are:')
    for k, v in ranked_models.items():
        trimmed_name = k[:-15]  # '#_metadata.json' = 15 chars
        if trimmed_name not in seen_configs:
            seen_configs.add(trimmed_name)
            unique_models[k] = v
            print(k[24:])  # this leaves out the path: 'models/saved/cv_results/'
            i += 1
        if i == n:
            break
    return unique_models


def get_metadata_hyperparams(unique_models):
    '''
    TODO: add description
    NOTE: we read from the metadata file so we can delete all cv models!
    '''
    hyperparams = []
    for path in list(unique_models.keys()):
        with open(path, 'r') as f:
            data = json.load(f)
        hyperparams.append(data['hyperparameters'])
    return hyperparams


def train_top_models():
    '''
    This is basically a one-time use function.
    GOAL: train the top 5 models and save the history, hyperparams, and model to the dir.
    *********
    read the metadata files and parse the performance metrics of each cv model.
    create a new dict with the path to the model as the key and {precision, f1, accuracy}
    as the value. sort these metrics in descending order to isolate the top n models.
    ensure that each n model is unique (recall that 4 models with the same config are saved for every model),
    we only want to add one of those to the top n list.
    train the top n models, save the metrics.
    '''
    image_list, labels = get_images()
    X_train, _, y_train, _ = split_data_train_test(image_list, labels)
    metadata_dict = get_metrics()
    top_n_models = get_top_n_models(metadata_dict)  # get top n models, currently 5
    hyperparameters = get_metadata_hyperparams(top_n_models)

    from src.train_full import train_full  # avoid circular imports with train_full
    for hyperparams in hyperparameters:
        train_full(X_train, y_train, hyperparams)
    return

def get_trained_model_paths(dir_path='models/saved/fully_trained'):
    '''
    get all the model paths from the default dir and add to a list.
    return list.
    '''
    pass


def save_metrics(test_metrics, dir_path='models/saved/fully_trained'):
    pass
    # metadata_dict = {}

    # with open(file_path, 'r') as f:
    #     data = json.load(f)
