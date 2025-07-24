from tensorflow import keras


def load_model(model_path):
    '''
    load a saved model
    '''
    try:
        model = keras.models.load_model(model_path)
        print(f'Model loaded successfully from {model_path}')
        return model
    except Exception as e:
        print(f'Error loading model from {model_path}: {e}')
        return
