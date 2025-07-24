import numpy as np


def prepare_tile_for_cnn(tile, augment=False):
    '''
    Prepare tile for CNN input: scale and optionally augment.
    Returns tile in shape (height, width, 1) for CNN.
    '''

    tile_normalized = tile.astype(np.float32) / 255.  # scale to 0-1 range
    tile_cnn = np.expand_dims(tile_normalized, axis=-1)  # add channel dim for CNN

    if augment:  # use me for training
        if np.random.random() > 0.7:  # random rotation (0, 90, 180, 270 deg)
            k = np.random.randint(0, 4)
            tile_cnn = np.rot90(tile_cnn, k=k, axes=(0, 1))

        if np.random.random() > 0.5:  # rand horizontal flip
            tile_cnn = np.flip(tile_cnn, axis=1)

    return tile_cnn
