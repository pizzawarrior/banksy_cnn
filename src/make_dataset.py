import numpy as np
from src.image_tiler import make_img_tiles
from src.get_entropy import get_img_entropy
from src.prepare_tiles import prepare_tile_for_cnn


def create_tiles_dataset(images, labels, tile_h, tile_w, overlap, entropy_threshold, augment=False):
    '''
    create dataset of tiles from images that pass entropy threshold.
    returns tiles in CNN format (batch, height, width, channels).
    '''
    tile_data = []
    tile_labels = []

    for img, label in zip(images, labels):
        tiles = make_img_tiles(img, tile_h, tile_w, overlap)
        img_entropy = get_img_entropy(img)

        assert img_entropy < entropy_threshold, f'Please use an entropy threshold lower than {img_entropy}'

        valid_tiles = 0
        for tile in tiles:
            tile_entropy = get_img_entropy(tile)
            if tile_entropy >= img_entropy - entropy_threshold:
                tile_cnn = prepare_tile_for_cnn(tile, augment=augment)
                tile_data.append(tile_cnn)
                tile_labels.append(label)
                valid_tiles += 1

        print(f'Image with label {label}: {valid_tiles}/{len(tiles)} tiles passed entropy threshold')

    return np.array(tile_data), np.array(tile_labels)
