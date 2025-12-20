import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay
                             )

from src.utils import save_plot

# TODO: should these fn's be moved to utils.py????

def get_metrics(y_test, y_pred, y_pred_proba, data_object, single_img=False):
    '''
    calculate comprehensive classification metrics
    data_object == 'tile' or 'img'.
    works for testing a single_img or a full test dataset.
    returns a dict with performance metrics.
    if testing on full dataset we also include the roc_auc_score.
    '''

    # for tracking training performance
    if data_object not in ['img', 'tile']:
        raise ValueError('Data object for calculating performance metrics must be either `tile` or `img`.')

    metric_funcs = {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
    }

    metrics = {
        name: func(y_test, y_pred)
        for name, func in metric_funcs.items()
    }

    if single_img is False:
        metrics.update({'auc': roc_auc_score(y_test, y_pred_proba)})

    return {data_object + '_' + k: v for k, v in metrics.items()}


def show_conf_matrix(y_test, y_pred, cm_path_model_name, save=False):
    '''
    create and display conf matrix for testing
    '''
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax)
    ax.set_title(cm_path_model_name[1])

    if save is True:
        save_plot('confusion matrix', cm_path_model_name)

    plt.show()
    plt.close(fig)  # prevent ghost plots


def show_training_accuracy_plots(history, acc_path_model_name):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(acc_path_model_name[1])
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    save_plot('accuracy plot', acc_path_model_name)
