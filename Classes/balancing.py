from numpy import unique, array, ndarray
from imblearn.over_sampling import SMOTE, RandomOverSampler


class BalanBasic:
    def __init__(self, balan_class: any, param: dict) -> None:
        self.__algorithm = balan_class(**param)
        self.__X = array([])
        self.__y = array([])

    def __repr__(self):
        ArrUniCounter = lambda x: dict(zip(unique(x, return_counts=True)))
        RatioCounter = lambda x, y: round(y[x] / sum(y.values()), 2)

        dist_dict = ArrUniCounter(self.__y)
        neg_num, neg_ratio = dist_dict[0], RatioCounter(0, dist_dict)
        pos_num, pos_ratio = dist_dict[1], RatioCounter(1, dist_dict)

        data_dist = 'Neg:Pos = {0}:{1}, ratio = {2}:{3}'.format(
            neg_num, pos_num, neg_ratio, pos_ratio)
        return data_dist

    def DataInit(self, X: any, y: ndarray) -> None:
        '''
        Initializing the dataset for Resampling
        X: feature values
        y: label values
        '''
        self.__X = X
        self.__y = y

    def OverSample(self) -> list:
        '''
        Resampling nach Algorithmus
        '''
        X_os, y_os = self.__algorithm.fit_resample(self.__X, self.__y)
        return X_os, y_os


class BalanSMOTE(BalanBasic):
    def __init__(self, X, y, s_param={'random_state': 0}) -> None:
        super().__init__(SMOTE, s_param)
        self.DataInit(X=X, y=y)


class BalanRandOS(BalanBasic):
    def __init__(self, X, y, s_param={'random_state': 0}) -> None:
        super().__init__(RandomOverSampler, s_param)
        self.DataInit(X=X, y=y)