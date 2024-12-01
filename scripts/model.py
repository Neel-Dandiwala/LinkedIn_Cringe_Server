import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from .extractor import Extractor
from .preprocessor import Preprocessor
import torch
from tqdm import tqdm

class ModelTrainer:
    def __init__(self):

        self.lgb_model = None
        self.extractor = Extractor()
        self.preprocessor = Preprocessor()
        self.feature_columns = None

    def prepare_features(self, texts, precomputed_features=None):
        bert_features = []
        traditional_features = []

        for i, text in enumerate(tqdm(texts)):
            bert_features.append(self.extractor.extract_bert_features(text))
            if precomputed_features is None:
                t_features = self.preprocessor.extract_features(text)
                traditional_features.append([t_features[f] for f in self.feature_columns])

        X_bert = np.array(bert_features)

        if precomputed_features is None:
            X_traditional = np.array(traditional_features)
        else:
            X_traditional = precomputed_features.values

        return np.hstack([X_bert, X_traditional])

    def train(self, df):
        """Training the model"""
        
        print("Extracting features from transformers")
        self.feature_columns = [col for col in df.columns if col not in ["cringe_score", "text"]]

        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        X_train = self.prepare_features(train_df["text"], train_df[self.feature_columns])
        y_train = train_df["cringe_score"].values

        X_val = self.prepare_features(val_df["text"], val_df[self.feature_columns])
        y_val = val_df["cringe_score"].values

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        print("Training LightGBM model")
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        self.lgb_model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[val_data])    

        importance = pd.DataFrame({
            'feature': ['bert_' + str(i) for i in range(X_train.shape[1] - len(self.feature_columns))] + self.feature_columns,
            'importance': self.lgb_model.feature_importance()
        })

        importance.to_csv("data/feature_importance.csv", index=False)

        print("Training complete")

        return self.lgb_model
        
    def save_model(self, path="models"):
        """Save the best model to a file"""
        if self.lgb_model is None:
            raise ValueError("No model to save")
        
        os.makedirs(path, exist_ok=True)

        model_data = {
            "model": self.lgb_model,
            "feature_columns": self.feature_columns
        }

        with open(f"{path}/model.pkl", "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {path}")
        
    
def main():
    df = pd.read_csv("data/processed_features.csv")

    trainer = ModelTrainer()
    trainer.train(df)

    trainer.save_model()


if __name__ == "__main__":
    main()


