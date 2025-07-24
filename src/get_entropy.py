import numpy as np
from scipy.stats import entropy


def get_img_entropy(img):
    '''
    flattens a grayscale image and tile then computes a histogram and probability dist of
    each pixel. then computes the total image entropy and returns it.
    '''
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    prob_dist = hist / hist.sum()
    img_ent = entropy(prob_dist, base=2)
    return img_ent
