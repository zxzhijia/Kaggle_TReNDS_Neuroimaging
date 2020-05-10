import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, config):
        self.config = config
        BASE_PATH = self.config['base_path']
        # load data here
        self.fnc_df = pd.read_csv(f"{BASE_PATH}/fnc.csv")
        self.loading_df = pd.read_csv(f"{BASE_PATH}/loading.csv")
        self.labels_df = pd.read_csv(f"{BASE_PATH}/train_scores.csv")
