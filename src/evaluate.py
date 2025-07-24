import numpy as np
from src.model_loader import load_model
from src.image_tiler import make_img_tiles
from src.get_entropy import get_img_entropy
from src.prepare_tiles import prepare_tile_for_cnn
from src.get_metrics import calc_metrics


def evaluate_on_test_set(X_test, y_test, hyperparams, model_path=None, model=None):
    '''
    Evaluate a trained model on the test set using tile-level predictions
    averaged to image-level predictions.
    '''
    if model is None and model_path is not None:
        model = load_model(model_path)
    elif model is None:
        raise ValueError('Either model or model_path must be provided')

    print('\nEvaluating on test set...')

    image_predictions = []
    image_true_labels = []

    for img, true_label in zip(X_test, y_test):
        tiles = make_img_tiles(img, hyperparams['tile_h'], hyperparams['tile_w'], hyperparams['overlap'])
        img_entropy = get_img_entropy(img)

        tile_predictions = []
        for tile in tiles:
            tile_entropy = get_img_entropy(tile)
            if tile_entropy >= img_entropy - hyperparams['entropy_threshold']:
                tile_cnn = prepare_tile_for_cnn(tile, augment=False)
                tile_cnn = np.expand_dims(tile_cnn, axis=0)  # add batch dimension
                pred = model.predict(tile_cnn, verbose=0)[0][0]
                tile_predictions.append(pred)

        if tile_predictions:
            # average tile predictions for final image prediction
            image_pred = np.mean(tile_predictions)
            image_predictions.append(image_pred)
            image_true_labels.append(true_label)
        else:
            print(f'Warning: No valid tiles found for test image with label {true_label}')

    # calc image-level metrics
    image_pred_binary = (np.array(image_predictions) > 0.5).astype(int)
    test_metrics = calc_metrics(
        image_true_labels, image_pred_binary, image_predictions
    )

    print('\nTest Set Results (Image-level):')
    for metric, value in test_metrics.items():
        print(f'{metric}: {value:.4f}')

    return test_metrics, image_predictions, image_true_labels
