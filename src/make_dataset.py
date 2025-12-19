import numpy as np
from src.get_image_tiles import make_img_tiles
from src.get_entropy import get_entropy
from src.prepare_tiles import prepare_tile_for_cnn
from src.utils import make_random_generator


def create_tiles_dataset(images, labels, tile_h: int, tile_w: int, overlap: float, entropy_threshold: float, augment=False):
    '''
    create dataset of tiles from images that pass the entropy threshold.
    accepts:
        - image array: X_train for training, or X_val_tiles for validation in cv training
        - labels array: y_train for training, or y_val_tiles for validation in cv training
    returns:
        - tile_data: np.array of tiles in CNN format (batch, height, width, channels).
        - tile_labels: np.array of corresponding tile labels
        - tile-image-map, np.array that stores the index of each parent image for each of the
        child tiles that is added to the dataset. This is used for image-level metric tracking.
    '''
    tile_data = []
    tile_labels = []
    tile_to_image_map = []

    for img_idx, (img, label) in enumerate(zip(images, labels)):
        tiles = make_img_tiles(img, tile_h, tile_w, overlap)
        img_entropy = get_entropy(img)

        assert entropy_threshold < img_entropy, f'Please use an entropy threshold lower than {img_entropy}'

        valid_tiles = 0
        for tile in tiles:
            tile_entropy = get_entropy(tile)
            if tile_entropy >= img_entropy - entropy_threshold:
                tile_cnn = prepare_tile_for_cnn(tile, augment=augment)
                tile_data.append(tile_cnn)
                tile_labels.append(label)
                tile_to_image_map.append(img_idx)
                valid_tiles += 1

        # print(f'Image with label {label}: {valid_tiles}/{len(tiles)} tiles passed entropy threshold: {img_entropy}.')

    tile_data = np.array(tile_data)
    tile_labels = np.array(tile_labels)
    tile_to_image_map = np.array(tile_to_image_map)

    # verify class balance across tiles
    # TODO: consider flipping or rotating the dupe tiles
    positive_count = np.sum(tile_labels)  # grab 1's
    negative_count = len(tile_labels) - positive_count  # grab 0's
    print(f'Before balancing classes, class 1 = {positive_count}/{len(tile_labels)}')

    if positive_count != negative_count:
        rng = make_random_generator()
        minority_label = 1 if positive_count < negative_count else 0
        minority_idxs = np.where(tile_labels == minority_label)[0]
        n_samples = abs(positive_count - negative_count)
        rand_idxs = rng.choice(minority_idxs, size=n_samples, replace=True)

        tile_data = np.concat((tile_data, tile_data[rand_idxs]))
        tile_labels = np.concat((tile_labels, tile_labels[rand_idxs]))
        tile_to_image_map = np.concat((tile_to_image_map, tile_to_image_map[rand_idxs]))

    return tile_data, tile_labels, tile_to_image_map
