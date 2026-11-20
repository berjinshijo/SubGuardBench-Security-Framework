import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def chronological_split(df, timestamp_col, test_fraction=0.2, validation_fraction=0.1):
    if timestamp_col and timestamp_col in df.columns:
        df = df.sort_values(timestamp_col)
    n = len(df)
    train_end = int(n * (1 - test_fraction - validation_fraction))
    val_end = int(n * (1 - test_fraction))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

def _onehot():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)

class MultimodalPreprocessor:
    """Fit transformations on training data only; transform validation/test without refitting."""
    def __init__(self, numeric_cols, categorical_cols, text_col=None, text_features=128, top_k=100):
        self.numeric_cols, self.categorical_cols = numeric_cols, categorical_cols
        self.text_col, self.text_features, self.top_k = text_col, text_features, top_k
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.encoder = _onehot()
        self.vectorizer = HashingVectorizer(n_features=text_features, alternate_sign=False, norm="l2")
        self.selected_idx = None
        self.feature_names = None

    def _existing(self, df, cols): return [c for c in cols if c and c in df.columns]

    def fit_transform(self, train, y=None):
        nums = self._existing(train, self.numeric_cols)
        cats = self._existing(train, self.categorical_cols)
        blocks = []
        names = []
        if nums:
            x = self.imputer.fit_transform(train[nums].apply(pd.to_numeric, errors="coerce"))
            x = self.scaler.fit_transform(x)
            blocks.append(x); names += [f"num:{c}" for c in nums]
        if cats:
            x = self.encoder.fit_transform(train[cats].fillna("MISSING").astype(str))
            blocks.append(x); names += [f"cat:{i}" for i in range(x.shape[1])]
        if self.text_col and self.text_col in train.columns:
            x = self.vectorizer.transform(train[self.text_col].fillna("").astype(str)).toarray()
            blocks.append(x); names += [f"text_hash:{i}" for i in range(x.shape[1])]
        if not blocks: raise ValueError("No configured numeric, categorical, or text columns were found. Edit config.yaml.")
        X = np.concatenate(blocks, axis=1).astype(np.float32)
        if y is not None and len(np.unique(y)) > 1:
            k = min(self.top_k, X.shape[1])
            scores = mutual_info_classif(X, y, random_state=42)
            self.selected_idx = np.argsort(scores)[-k:]
        else:
            self.selected_idx = np.arange(min(self.top_k, X.shape[1]))
        self.feature_names = names
        return X[:, self.selected_idx]

    def transform(self, df):
        nums = self._existing(df, self.numeric_cols)
        cats = self._existing(df, self.categorical_cols)
        blocks = []
        if nums:
            x = self.imputer.transform(df[nums].apply(pd.to_numeric, errors="coerce"))
            blocks.append(self.scaler.transform(x))
        if cats:
            blocks.append(self.encoder.transform(df[cats].fillna("MISSING").astype(str)))
        if self.text_col and self.text_col in df.columns:
            blocks.append(self.vectorizer.transform(df[self.text_col].fillna("").astype(str)).toarray())
        if not blocks: raise ValueError("No usable configured feature columns in this split.")
        return np.concatenate(blocks, axis=1).astype(np.float32)[:, self.selected_idx]
