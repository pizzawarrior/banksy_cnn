import os
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt

'''
we want to read in all images and add them to a single matrix.
will need to immediately create corresponding labels.
then we use np.random and training-test-split to pick training and testing images.
labels are adjusted according to the split.
returns a matrix of training images and training labels, and a matrix of testing images and testing labels
import ALL images.
convert to grayscale.
'''


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


def get_images(img_dir='images'):
    '''
    loads all images, saves them to a list.
    also creates a list of labels and returns it.
    0 indicates 'not banksy', 1 indicates 'banksy'.
    '''
    file_names = os.listdir(img_dir)
    img_files = [f for f in file_names if '.jpg' in f]
    img_list = []
    labels = [0 if 'bkf' in f else 1 for f in img_files]  # create binary list of labels

    for img_file in img_files:
        img = load_image(img_file, img_dir)  # these are NOT loaded in the order shown in the images dir.
        img_list.append(np.array(img))
    print('Images loaded')
    return img_list, labels
