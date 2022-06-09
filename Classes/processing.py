import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


class Basic:
    def __init__(self) -> None:
        pass


class DataSetProcess(Basic):
    def __init__(self, data_: pd.DataFrame, col_label: str):
        super().__init__()
        self.__data = data_
        self.__coln = col_label

    def __GetXy(self):
        data_ = self.__data
        X = data_.loc[:, data_.columns != self.__coln]
        y = data_.loc[:, data_.columns == self.__coln].values.ravel()
        return X, y

    def DataImpute(self, impute_type: str = 'knn'):
        data_ = self.__data

        if impute_type == 'knn':
            imp = KNNImputer(weights='uniform')

        data_imp = imp.fit_transform(data_.values.tolist())
        for i in range(data_.shape[0]):
            data_.loc[data_.index[i]] = data_imp[i]

    def DataSplit(self, test_size_st: float):
        X, y = self.__GetXy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_st, random_state=0, stratify=y)
        return X_train, y_train, X_test, y_test

    def KFoldSplit(self, split_n: int, rand_n: int = 0):
        data_l = []
        X, y = self.__GetXy()
        _, counts = np.unique(y, return_counts=True)
        y_invalid = [count < split_n for count in counts]

        if sum(y_invalid) > 0:
            return data_l

        kf = StratifiedKFold(n_splits=split_n,
                             random_state=rand_n,
                             shuffle=True if rand_n > 0 else False)

        for train_i, test_i in kf.split(X, y):
            X_train, y_train = X.iloc[train_i], y[train_i]
            X_test, y_test = X.iloc[test_i], y[test_i]
            data_l.append([X_train, y_train, X_test, y_test])

        return data_l


class OutcomeGenerate(Basic):
    def __init__(self) -> None:
        super().__init__()
