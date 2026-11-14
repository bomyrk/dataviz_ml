# src/modeling/create_folds.py
import numpy as np
import pandas as pd
from . import config
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold

def create_folds(data, type_prob: str = "chronological", n_splits: int = 5):
    """
    Create folds on either chronological or regular tabular data (Train only) 
    Returns train with a 'kfold' column (the fold index each row belongs to as validation).
    """
    data = data.copy()
    # we add a new column call kfold and fill it with -1
    data["kfold"] = -1
    
    if type_prob == "chronological":
        data = data.sort_values('Date').reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for f, (t_, v_) in enumerate(tscv.split(data)):
            # for each row   gets the fold number where it serves as validation
            data.loc[v_, 'kfold'] = f
    elif type_prob == "classification":
        # randomize the rows of the data
        data = data.sample(frac=1).reset_index(drop=True)
        # initiate the kfold class from model_selection module
        kf = StratifiedKFold(n_splits=n_splits)
        # fill the new kfold column
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data.label.values)):
            data.loc[v_, 'kfold'] = f
        
    elif type_prob == "regression":
        data = data.sample(frac=1).reset_index(drop=True)
        num_bins = int(np.floor(1 + np.log2(len(data))))
        data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False)
        kf = StratifiedKFold(n_splits=n_splits)
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
            data.loc[v_, 'kfold'] = f
        # drop the bins column
        data.drop("bins", axis=1, inplace=True)

    else:
        raise ValueError(f"Error problem type: {type_prob} unknown")
        
    # return dataframe with folds
    return data

if __name__ == "__main__":
    train = pd.read_csv(config.TRAIN_FILE, parse_dates=['Date'])
    # create folds
    train = create_folds(train, config.TYPE_PROB, config.NBR_SPLITS)
    train.to_csv(config.TRAIN_FOLDS_FILE, index=False)
    print(f"Folds created. kfold values: {sorted(train['kfold'].unique())}")
    print(train['kfold'].value_counts().sort_index())
