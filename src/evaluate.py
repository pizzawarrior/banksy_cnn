import numpy as np
from src.model_loader import load_model
from src.image_tiler import make_img_tiles
from src.get_entropy import get_img_entropy
from src.prepare_tiles import prepare_tile_for_cnn
from src.get_metrics import calc_metrics, show_conf_matrix
from src.get_hyperparams import get_hyperparams


def evaluate_on_test_set(X_test, y_test, hyperparams=None, model_path=None, model=None):
    '''
    Evaluate a trained model on the test set using tile-level predictions
    averaged to image-level predictions.
    '''
    if model is None and model_path is not None:
        model = load_model(model_path)
    elif model is None:
        raise ValueError('Either model or model_path must be provided')

    if model_path:
        hyperparams = get_hyperparams(model_path)

    print('\nEvaluating on test set...')

    image_predictions = []
    image_true_labels = []

    tiles_attempted = 0
    total_tiles_used = 0
    image_num = 1

    for img, true_label in zip(X_test, y_test):
        tiles = make_img_tiles(img, hyperparams['tile_h'], hyperparams['tile_w'], hyperparams['overlap'])
        img_entropy = get_img_entropy(img)

        tile_predictions = []
        for tile in tiles:
            tile_entropy = get_img_entropy(tile)
            tiles_attempted += 1
            if tile_entropy >= img_entropy - hyperparams['entropy_threshold']:
                tile_cnn = prepare_tile_for_cnn(tile, augment=False)
                tile_cnn = np.expand_dims(tile_cnn, axis=0)  # add batch dim
                pred = model.predict(tile_cnn, verbose=0)[0][0]
                tile_predictions.append(pred)
                total_tiles_used += 1

        print(f'Test Image {image_num}: There were {total_tiles_used}/{tiles_attempted} for testing')
        image_num += 1

        if tile_predictions:
            # average tile predictions for final image classification
            image_pred = np.mean(tile_predictions)
            image_predictions.append(image_pred)
            image_true_labels.append(true_label)
        else:
            print(f'Warning: No valid tiles found for test image with label {true_label}')

    # calc image-level metrics
    classification_threshold = hyperparams.get('classification_threshold', .4)
    image_pred_binary = (np.array(image_predictions) > classification_threshold).astype(int)
    test_metrics = calc_metrics(
        image_true_labels, image_pred_binary, image_predictions, 'img'
    )

    print('\nTest Set Results (Image-level):')
    print(f'classification-threshold: {classification_threshold}')
    for metric, value in test_metrics.items():
        print(f'{metric}: {value:.4f}')

    show_conf_matrix(image_true_labels, image_pred_binary)

    # experimenting ONLY; use for single image testing
    # return test_metrics, image_predictions, image_true_labels, tile_predictions

    return test_metrics, image_predictions, image_true_labels
