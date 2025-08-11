from models.architectures.cnn_3_layer import cnn_3_layer
from models.architectures.cnn_5_layer import cnn_5_layer


def model_builders(hyperparams):
    model_builders = {
    '3layer': cnn_3_layer,
    '5layer': cnn_5_layer,
    }

    try:
        builder = model_builders[hyperparams['architecture']]
        model = builder(hyperparams['tile_h'], hyperparams['tile_w'], hyperparams['learning_rate'])
    except KeyError:
        raise ValueError(
            f"Model name '{hyperparams['architecture']}' is invalid. Must be one of: {', '.join(model_builders.keys())}."
        )

    return model
