import tensorflow as tf


def configure_tf():
    '''
    configure tf to use available local GPU via tensorflow-metal library.
    memory growth must be set before GPUs have been initialized.
    also sets tf global precision to mixed float 16
    '''
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    return
