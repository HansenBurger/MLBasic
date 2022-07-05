import numpy as np
from pandas import Series


class Basic():
    def __init__(self) -> None:
        pass


class DataSets(Basic):
    def __init__(self) -> None:
        super().__init__()
        self.__X_train = np.array([])
        self.__y_train = np.array([])
        self.__X_test = np.array([])
        self.__y_test = np.array([])

    @property
    def X_train(self):
        return self.__X_train

    @X_train.setter
    def X_train(self, v):
        self.__X_train = v

    @property
    def y_train(self):
        return self.__y_train

    @y_train.setter
    def y_train(self, v):
        self.__y_train = v

    @property
    def X_test(self):
        return self.__X_test

    @X_test.setter
    def X_test(self, v):
        self.__X_test = v

    @property
    def y_test(self):
        return self.__y_test

    @y_test.setter
    def y_test(self, v):
        self.__y_test = v


class Predictions(Basic):
    def __init__(self) -> None:
        super().__init__()
        self.__a_fpr = np.array([])
        self.__a_tpr = np.array([])
        self.__s_sen = -1.0
        self.__s_spe = -1.0
        self.__s_acc = -1.0
        self.__s_f_1 = -1.0
        self.__s_r_2 = -1.0
        self.__s_auc = -1.0
        self.__report = {}
        self.__ft_imp = Series([])

    @property
    def a_fpr(self):
        return self.__a_fpr

    @a_fpr.setter
    def a_fpr(self, v: np.ndarray):
        self.__a_fpr = v

    @property
    def a_tpr(self):
        return self.__a_tpr

    @a_tpr.setter
    def a_tpr(self, v: np.ndarray):
        self.__a_tpr = v

    @property
    def s_sen(self):
        return self.__s_sen

    @s_sen.setter
    def s_sen(self, v: float):
        self.__s_sen = v

    @property
    def s_spe(self):
        return self.__s_spe

    @s_spe.setter
    def s_spe(self, v: float):
        self.__s_spe = v

    @property
    def s_acc(self):
        return self.__s_acc

    @s_acc.setter
    def s_acc(self, v: float):
        self.__s_acc = v

    @property
    def s_f_1(self):
        return self.__s_f_1

    @s_f_1.setter
    def s_f_1(self, v: float):
        self.__s_f_1 = v

    @property
    def s_r_2(self):
        return self.__s_r_2

    @s_r_2.setter
    def s_r_2(self, v: float):
        self.__s_r_2 = v

    @property
    def s_auc(self):
        return self.__s_auc

    @s_auc.setter
    def s_auc(self, v: float):
        self.__s_auc = v

    @property
    def report(self):
        return self.__report

    @report.setter
    def report(self, v: dict):
        self.__report = v

    @property
    def ft_imp(self):
        return self.__ft_imp

    @ft_imp.setter
    def ft_imp(self, v: any):
        self.__ft_imp = v