import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay
                             )


def get_metrics(y_test, y_pred, y_pred_proba, data_object, single_img=False):
    '''
    calculate comprehensive classification metrics
    data_object == 'tile' or 'img'
    '''

    # for tracking training performance
    if data_object not in ['img', 'tile']:
        raise ValueError('Data object for calculating performance metrics must be either `tile` or `img`.')

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
    }

    if single_img is False:
        metrics.update({'auc': roc_auc_score(y_test, y_pred_proba)})

    return {data_object + '_' + k: v for k, v in metrics.items()}


def show_conf_matrix(y_test, y_pred):
    '''
    create and display conf matrix for testing
    '''
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()
