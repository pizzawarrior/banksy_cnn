import json
import re


def get_hyperparams(model_path):
    assert isinstance(model_path, str), 'Model path must be a string'

    # extract the filename without the extension
    pattern = r'(.*?)(?:\.keras|\.h5)$'
    match = re.match(pattern, model_path)

    if not match:
        raise ValueError('Model path must be of type .keras or .h5')

    json_path = match.group(1) + '_metadata.json'  # build new path to locate json file
    with open(json_path, 'r') as f:
        config = json.load(f)

    hyperparams = config.get('hyperparameters', None)
    return hyperparams
