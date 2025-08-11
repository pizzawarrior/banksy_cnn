import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from src.data_loader import get_images
from src.split_data import split_data_train_test
from src.make_dataset import create_tiles_dataset
from src.save_model import save_model_with_metadata
from src.evaluate import get_metrics
from models.architectures.cnn_3_layer import cnn_3_layer
from models.architectures.cnn_5_layer import cnn_5_layer


def train_full(X_train,
                y_train,
                hyperparams,
                save_dir='models/saved/fully_trained'):
    '''
    search thru all metadata files and find the best performing models.
    load the hyperparams of the top 5 UNIQUE model configurations.
    (prevent multiple folds from the same model config from counting as different models).
    train those 5 models and save their metrics.
    TODO: double check the training/ validation metrics that we have access to and can save.
    '''
    pass
