import json


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
