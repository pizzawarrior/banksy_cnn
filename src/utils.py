import json
import os
import re
from tensorflow import keras
from models.architectures.cnn_3_layer import cnn_3_layer
from models.architectures.cnn_5_layer import cnn_5_layer
from src.data_loader import get_images
from src.split_data import split_data_train_test


def load_model(model_path):
    '''
    load a saved model using the provided model path.
    this is used for evaluating a model on test data.
    '''
    try:
        model = keras.models.load_model(model_path)
        print(f'Model loaded successfully from {model_path}')
        return model
    except Exception as e:
        print(f'Error loading model from {model_path}: {e}')
        return


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


def get_metadata(file_path):
    '''
    take metadata file path and parse the json file.
    grab precision, f1, and accuracy from the 'metrics' key.
    return a dict of model performance metrics.
    NOTE: early on there was an error in the way training and validation metrics
        were recorded, and all the validation metrics were actually a copy of the training ones.
        This was later changed to tile_*metric* and img_*metric*. Because earlier models only
        recorded training metrics, we compare training metrics, not validation ones.
        This should be changed in the future as new models are added, and the older ones are removed.
    '''
    assert isinstance(file_path, str), 'Metadata file path must be a string'

    with open(file_path, 'r') as f:
        config = json.load(f)

    metrics = config.get('metrics')
    results = {}

    if metrics:
        output_metrics = ['precision', 'f1', 'accuracy']
        for metric in output_metrics:
            value = metrics.get(metric)
            if value is None:
                # see above for why we don't use img_*metric*
                value = metrics.get(f'tile_{metric}')
            results[metric] = round(value, 4)

    else:
        print(f'No metadata found for file path: {file_path}')

    return results


def get_saved_metrics(dir_path='models/saved/cv_results'):
    '''
    reads the metadata files from cv_results and parse the performance metrics of each cv model.
    create a new dict with the path to the model as the key and {precision, f1, accuracy}
    as the value.
    returns the dict.
    this is used to locate the best models from the cv_results. likely a single-use function.
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
    take a dictionary of metadata performance metrics and sort them in
    descending order to isolate the top n models.
    ensure that each n model is unique (recall that 4 models with the same config
    are saved for every model), we only want to add one of those to the top n list.
    this is used to locate the best models from the cv_results. likely a single-use function.
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
    collect the hyperparameters for the top n models so we can train new
    models with the full dataset using these hyperparams.
    returns a list of dictionaries of hyperparameters for each model.
    this is used to locate the best models from the cv_results. likely a single-use function.
    TODO: we read from the metadata file so we can delete all cv models!
    '''
    hyperparams = []
    for path in list(unique_models.keys()):
        with open(path, 'r') as f:
            data = json.load(f)
        hyperparams.append(data['hyperparameters'])
    return hyperparams


def train_top_models():
    '''
    train the top n models, save the metrics.
    this is basically a throwaway function.
    '''
    image_list, labels = get_images()
    X_train, _, y_train, _ = split_data_train_test(image_list, labels)
    metadata_dict = get_saved_metrics()
    top_n_models = get_top_n_models(metadata_dict)  # get top n models, currently 5
    hyperparameters = get_metadata_hyperparams(top_n_models)

    from src.train_full import train_full  # avoid circular imports with train_full
    for hyperparams in hyperparameters:
        train_full(X_train, y_train, hyperparams)
    return


def get_trained_model_paths(dir_path='models/saved/fully_trained'):
    '''
    get all the model paths from the default dir and add to a list.
    return list. we gather all of them so we can test them.
    this may be a one-time-use function.
    '''
    model_paths = []

    # find all metadata paths, parse the files, and add the metrics to a dict
    for dir_path, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith('metadata.json'):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                model_path = data.get('model_path')
                if model_path is None:
                    raise ValueError(f'No model path can be found in {file_path}')
                model_paths.append(model_path)
    return model_paths


def save_metrics(test_metrics, model_path):
    '''
    parse the provided model_path and format it so we can update the metadata file
    for the given model and add test metrics. all existing data is retained and untouched.
    the metadata file is then saved with this new data.
    NOTE: the key used here for metrics is different than in cv_results. Here we use 'test_metrics'.
    '''
    assert '.keras' in model_path, f'Provided model path: {model_path} is in the incorrect format and can not be parsed for file saving.'
    metadata_path = model_path.replace('.keras', '_metadata.json')
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'Error: the metadata file path: {metadata_path} was not found.')
        exit()
    data['test_metrics'] = test_metrics
    with open(metadata_path, 'w') as f:
        json.dump(data, f, indent=2)
    print('Model metadata file updated successfully with test-metrics.')


def get_model_name(model_path):
    '''
    take a relative model path, strip the path, and keep only the model name.
    returns a string for the full model name.
    this is used to create a title for the confusion matrix when a model is
    tested.
    '''
    pattern = r'.*\/(.*)'
    match = re.match(pattern, model_path)
    if not match:
        raise ValueError('Error trying to parse model path')
    model_name = match.group(1)
    return model_name


def make_training_plots():
    '''
    make and save loss and accuracy plots for the saved fully trained models
    '''
    pass
