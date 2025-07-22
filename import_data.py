import os
from PIL import Image
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

'''
we want to read in all images and add them to a single matrix.
will need to immediately create corresponding labels.
then we use np.random and training-test-split to pick training and testing images.
labels are adjusted according to the split.
returns a matrix of training images and training labels, and a matrix of testing images and testing labels

import ALL images.
convert to grayscale and vectorize each one.
'''


def get_img_entropy(img):
    '''
    flattens a grayscale image then computes a histogram and probability dist of
    each pixel. then computes the total image entropy and returns it.
    '''
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    prob_dist = hist / hist.sum()
    img_ent = entropy(prob_dist, base=2)
    return img_ent


def split_data_train_test(dataset_length):
    '''
    takes the length of the full dataset and return random
    indices for training and testing sets.
    ~80% train, 20% test = 44 train, 12 test
    '''
    train_count = int(dataset_length * .8)
    seed = 420
    rng = np.random.default_rng(seed)
    train_indices = rng.choice(a=dataset_length, size=train_count, replace=False)
    test_indices = [num for num in range(dataset_length) if num not in train_indices]
    return train_indices, test_indices


def make_img_tiles(img, h, w, overlap):
    '''
    take an image, and break it into as many tiles as possible
    using the provided params.
    overlap must be a percent of type float.
    NOTE: ALL TILES MUST BE THE SAME EXACT SIZE, NO EXCEPTIONS.
    '''

    def _verify_tile_size(tile):
        if tile.shape[1] != w and tile.shape[0] != h:
            tile = img[-h:, -w:]
        elif tile.shape[0] != h:
            tile = img[-h:, j:j + w]
        elif tile.shape[1] != w:
            tile = img[i:i + h, -w:]
        return tile

    tiles = []
    orig_height, orig_len = img.shape

    assert isinstance(overlap, float)
    offset = 1 - overlap  #Ex: overlap=80, h=100, w=100, offset=20
    overlap_h, overlap_w = int(offset * h), int(offset * w)

    # generic tiles, minimal overlap
    for i in range(0, orig_height, h):
        for j in range(0, orig_len, w):
            tile = img[i:i + h, j:j + w]
            tiles.append(_verify_tile_size(tile))
            tile_overlap = img[i + overlap_h:i + overlap_h + h, j + overlap_w:j + overlap_w + w]
            tiles.append(_verify_tile_size(tile_overlap))

    return tiles


def make_dataset(tiles, h=100, w=100, overlap=.5):
    '''
    flatten all tiles that pass the entropy test, and add them to a single np.array (matrix).
    this is for training and testing.
    normalize both matrices by dividing by 255.
    make new labels vector.
    '''
    pass


def convert_tile_to_vector(img):
    '''
    flatten each TILE and turn it into a vector
    '''
    img_vec = np.array(img).reshape(-1)
    return img_vec


def load_image(img_file, img_dir):
    '''
    loads a single image as a PIL image object and returns it
    '''
    img_path = os.path.join(img_dir, img_file)
    try:
        img = Image.open(img_path).convert('L')  # convert to grayscale
    except IOError:
        print(f'Could not open or process image: {img_file}')
    return img


def preprocess_images(img_dir='images'):
    '''
    loads all images, converts them to grayscale and saves them to a python list.
    also creates a list of labels and returns the list of images with respective labels.
    0 indicates 'not banksy', 1 indicates 'banksy'.
    '''
    file_names = os.listdir(img_dir)
    # print(file_names)
    img_files = [f for f in file_names if '.jpg' in f]
    img_list = []
    labels = [0 if 'bkf' in f else 1 for f in img_files]  # create binary labels list

    for img_file in img_files:
        img = load_image(img_file, img_dir)  # these are NOT loaded in the order shown in the images dir.
        img_list.append(np.array(img))

    return img_list, labels


images_list, labels = preprocess_images()


img = images_list[9]
new_img = img[:542, :631]
# print(new_img.shape)
tiles = make_img_tiles(new_img, 100, 100, overlap=.8)
print(len(tiles))

# for tile in tiles:
#     plt.imshow(tile, cmap='gray')
#     plt.show()
# print([arr.shape for arr in tiles])
