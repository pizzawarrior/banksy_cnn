from src.data_loader import get_images
from src.train import train_model_with_cv
from src.split_data import split_data_train_test


def run_experiment():
    '''
    run a complete experiment with different hyperparameter sets FOR TRAINING.
    returns a list of dictionaries of model metadata and training metrics.
    '''
    image_list, labels = get_images('images')
    X_train, _, y_train, _ = split_data_train_test(image_list, labels)

    # define hyperparameter sets to test
    hyperparameter_sets = [
        {
            'tile_h': 200, 'tile_w': 200, 'overlap': 0.8, 'entropy_threshold': 1.0,
            'architecture': '5layer', 'learning_rate': 0.001
        },
        # {
        #     'tile_h': 100, 'tile_w': 100, 'overlap': 0.5, 'entropy_threshold': 2.0,
        #     'architecture': '5layer', 'learning_rate': 0.001
        # },
        # {
        #     'tile_h': 150, 'tile_w': 150, 'overlap': 0.8, 'entropy_threshold': 3.0,
        #     'architecture': '5layer', 'learning_rate': 0.001
        # }
    ]

    results = []

    for i, hyperparams in enumerate(hyperparameter_sets, 1):
        print(f'\n{"#"*60}')
        print(f'EXPERIMENT {i}/{len(hyperparameter_sets)}')
        print(f'{"#"*60}')

        # train model, may need to increase epochs/ batches
        cv_summary, fold_results, saved_models = train_model_with_cv(
            X_train, y_train, hyperparams, n_folds=4, epochs=60, batch_size=16
        )

        results.append({
            'hyperparams': hyperparams,
            'cv_summary': cv_summary,
            'fold_results': fold_results,
            'saved_models': saved_models
        })

    print(f'\n{"#"*60}')  # print final comparison
    print('FINAL COMPARISON')
    print(f'{"#"*60}')

    for i, result in enumerate(results, 1):
        hyperparams = result['hyperparams']
        cv_summary = result['cv_summary']
        print(f'\nExperiment {i}:')
        print(f'Tiles: {hyperparams["tile_h"]}x{hyperparams["tile_w"]}')
        print(f'Overlap: {hyperparams["overlap"]}, Entropy: {hyperparams["entropy_threshold"]}')
        print(f'Architecture: {hyperparams["architecture"]}')
        print(f'CV Accuracy: {cv_summary["accuracy_mean"]:.4f} ± {cv_summary["accuracy_std"]:.4f}')
        print(f'CV F1: {cv_summary["f1_mean"]:.4f} ± {cv_summary["f1_std"]:.4f}')

    return results
