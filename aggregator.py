# aggregator.py

import pandas as pd

def combine_ensemble_predictions(svm_preds, rf_preds, lstm_preds, rnn_preds):
    """
    Inputs: 4 DataFrames with shape (7, 2) — columns ['Predicted_Open', 'Predicted_Close']
    Returns: Ensemble-averaged prediction
    """
    combined = pd.concat(
        [svm_preds, rf_preds, lstm_preds, rnn_preds],
        axis=1
    )

    # Calculate row-wise mean of each type
    combined['Ensemble_Open'] = combined.iloc[:, [0, 2, 4, 6]].mean(axis=1)
    combined['Ensemble_Close'] = combined.iloc[:, [1, 3, 5, 7]].mean(axis=1)

    ensemble_df = combined[['Ensemble_Open', 'Ensemble_Close']]
    return ensemble_df
