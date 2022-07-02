import warnings
import pandas as pd

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report, confusion_matrix

from source_domain import DataSets, Predictions

warnings.filterwarnings('ignore')


class ModelBasic:
    def __init__(self, algorithm):
        self.__model = None
        self.__dataset = DataSets()
        self.__perform = Predictions()
        self.__algorithm = algorithm

    @property
    def dataset(self):
        return self.__dataset

    @property
    def perform(self):
        return self.__perform

    @property
    def algorithm(self):
        return self.__algorithm

    def DataInit(self, X_train, y_train, X_test, y_test) -> None:
        self.__dataset.X_train = X_train
        self.__dataset.y_train = y_train
        self.__dataset.X_test = X_test
        self.__dataset.y_test = y_test

    def ModelInit(self, param: dict) -> None:
        '''
        Initializing the model with model classes and parameters
        model_cls: any model class

        '''
        self.__model = self.__algorithm(**param)

    def Deduce(self, param: dict = {}) -> None:
        '''
        Use traning data to deduce the model
        '''
        X_in = self.__dataset.X_train
        y_in = self.__dataset.y_train

        self.__model.fit(X_in, y_in, **param)

    def Predict(self) -> dict:
        '''
        Use test data to estimate the model
        '''
        X_in = self.__dataset.X_test
        y_in = self.__dataset.y_test

        p_label = self.__model.predict(X_in)
        p_proba = self.__model.predict_proba(X_in)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_in, p_label, labels=[0, 1]).ravel()
        fpr, tpr, _ = roc_curve(y_in, p_proba)
        self.__perform.a_fpr = fpr
        self.__perform.a_tpr = tpr
        self.__perform.s_sen = tp / (tp + fn)
        self.__perform.s_spe = tn / (fp + tn)
        self.__perform.s_acc = (tp + tn) / (tp + tn + fp + fn)
        self.__perform.s_f_1 = tp / (tp + (fp + fn) / 2)
        self.__perform.s_r_2 = self.__model.score(X_in, y_in)
        self.__perform.s_auc = roc_auc_score(y_in, p_proba)
        self.__perform.report = classification_report(y_in, p_label)


class LogisiticReg(ModelBasic):
    def __init__(self, data_: list = [0, 0, 0, 0], s_param: dict = {}):
        super().__init__(LogisticRegression)
        self.DataInit(*data_)
        self.ModelInit(s_param)

    def GetFeatureImp(self) -> pd.Series:
        feat_name = self._ModelBasic__dataset.X_train.columns.tolist()
        feat_imp = self._ModelBasic__model.coef_[0]
        attr_imp = pd.Series(dict(zip(feat_name, feat_imp)))
        return attr_imp


class RandomForest(ModelBasic):
    def __init__(self, data_: list = [0, 0, 0, 0], s_param: dict = {}):
        super().__init__(RandomForestClassifier)
        self.DataInit(*data_)
        self.ModelInit(s_param)

    def GetFeatureImp(self) -> pd.Series:
        feat_name = self._ModelBasic__dataset.X_train.columns.tolist()
        feat_imp = self._ModelBasic__model.feature_importances_
        attr_imp = pd.Series(dict(zip(feat_name, feat_imp)))
        return attr_imp


class SupportVector(ModelBasic):
    def __init__(self, data_: list = [0, 0, 0, 0], s_param: dict = {}):
        super().__init__(SVC)
        self.DataInit(*data_)
        self.ModelInit(s_param)

    def GetFeatureImp(self) -> pd.Series:
        return pd.Series([])


class ParaSel_Grid(ModelBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__(GridSearchCV)
        self.DataInit(*data_)
        self.ModelInit(s_param)

    def BestParam(self) -> dict:
        best_param = self._ModelBasic__model.best_params_
        return best_param


class ParaSel_Rand(ModelBasic):
    def __init__(self, data_: list, s_param: dict = {}):
        super().__init__(RandomizedSearchCV)
        self.DataInit(*data_)
        self.ModelInit(s_param)

    def BestParam(self) -> dict:
        best_param = self._ModelBasic__model.best_params_
        return best_param


class XGBoosterClassify(ModelBasic):
    def __init__(self, data_: list = [0, 0, 0, 0], s_param: dict = {}):
        super().__init__(XGBClassifier)
        self.DataInit(*data_)
        self.ModelInit(s_param)

    def GetFeatureImp(self) -> pd.Series:
        feat_name = self._ModelBasic__dataset.X_train.columns.tolist()
        feat_imp = self._ModelBasic__model.feature_importances_
        attr_imp = pd.Series(dict(zip(feat_name, feat_imp)))
        return attr_imp

    def RebootByImpFeats(self) -> pd.Series:
        data_set = self._ModelBasic__dataset
        attr_imp = pd.Series(
            self._ModelBasic__model.get_booster().get_fscore())
        attr_select = attr_imp.loc[attr_imp > 0].index

        if attr_select.empty:
            pass
        else:
            self._ModelBasic__dataset.X_train = data_set.X_train[attr_select]
            self._ModelBasic__dataset.X_test = data_set.X_test[attr_select]