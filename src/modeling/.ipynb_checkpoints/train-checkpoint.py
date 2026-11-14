# src/modeling/train.py

import os
import config
import joblib
import argparse
import pandas as pd
import model_dispatcher as md
from sklearn import metrics

def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # training data is where kfold is not equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the target column from the dataframe and convert it to
    # numpy array by using .values method
    X_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    # simirlaly, for validation, we have
    X_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    # initialize simple decision tree classifier from sklearn
    clf = md.models[model]

    # fit the model on training data
    clf.fit(X_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(X_valid)

    #calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")


    # save the model
    joblib.dump(
        clf, 
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    # read the arguments from the command line
    args = parser.parse_args()
    
    # run the fold specified by command line arguments
    run(
        fold=args.fold, 
        model=args.model
    )
    #run(fold=0)
    #run(fold=1)
    #run(fold=2)
    #run(fold=3)
    #run(fold=4)