import tensorflow
import os
import random
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as pyplot
import pandas as pd
import sys
sys.path.append("../")
from data_loader.data_load import DataLoader
from utils.config import get_config_from_json
from models.svm_model import train_svm

def data_process(data):

    fnc_df = data.fnc_df
    loading_df = data.loading_df
    labels_df = data.labels_df

    fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
    df = fnc_df.merge(loading_df, on="Id")
    labels_df["is_train"] = True
    df = df.merge(labels_df, on="Id", how="left")

    train_df = df[df["is_train"] == True].copy()
    test_df = df[df["is_train"] != True].copy()
    target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
    test_df = test_df.drop(target_cols + ['is_train'], axis=1)
    # train_df = train_df.drop(target_cols + ['is_train'], axis=1)

    # Giving less importance to FNC features since they are easier to overfit due to high dimensionality.
    FNC_SCALE = 1/500
    test_df[fnc_features] *= FNC_SCALE
    train_df[fnc_features] *= FNC_SCALE
    target_models_dict = {
    'age': 'age_br',
    'domain1_var1':'domain1_var1_ridge',
    'domain1_var2':'domain1_var2_svm',
    'domain2_var1':'domain2_var1_ridge',
    'domain2_var2':'domain2_var2_svm',
    }

    return train_df, test_df, target_cols, loading_features, fnc_features

if __name__ == "__main__":
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        config_path = "../configs/example.json"
        config = get_config_from_json(config_path)

    except:
        print("missing or invalid arguments")
        exit(0)


    data_loader = DataLoader(config[1])
    train_df, test_df, target_cols, loading_features, fnc_features = data_process(data_loader)
    train_svm(config[1]['num_fold'], loading_features, fnc_features, train_df, test_df)
