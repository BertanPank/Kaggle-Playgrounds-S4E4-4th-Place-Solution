from autogluon.core.metrics import make_scorer
from sklearn.metrics import mean_squared_log_error
import numpy as np

def rmsle(y_true, y_pred):
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    return rmsle

rmsle_scorer = make_scorer(name='mean_squared_log_error',
                                 score_func=rmsle,
                                 optimum=0,
                                 greater_is_better=False)