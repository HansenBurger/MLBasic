import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
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

    def KFoldSplit(self, split_n: int, rand_n: int = None):
        data_l = []
        X, y = self.__GetXy()
        _, counts = np.unique(y, return_counts=True)
        y_invalid = [count < split_n for count in counts]

        if sum(y_invalid) > 0:
            return data_l

        kf = StratifiedKFold(n_splits=split_n,
                             random_state=rand_n,
                             shuffle=True if not rand_n else False)

        for train_i, test_i in kf.split(X, y):
            X_train, y_train = X.iloc[train_i], y[train_i]
            X_test, y_test = X.iloc[test_i], y[test_i]
            data_l.append([X_train, y_train, X_test, y_test])

        return data_l


class OutcomeGenerate(Basic):
    def __init__(self, perform: list, save_path: Path) -> None:
        super().__init__()
        self.__perform = perform
        self.__save_path = save_path

    def __SaveGen(self, save_name: str, save_suffix: str):
        self.__save_path.mkdir(parents=True, exist_ok=True)
        save_path = self.__save_path / '.'.join([save_name, save_suffix])
        return save_path

    def __DictToText(self, dict_: dict, l_gap: str = '\n', v_gap: str = ':\t'):
        repr_gen = lambda dict_: (l_gap).join(
            [v_gap.join([k, str(v)]) for k, v in dict_.items()])
        return repr_gen(dict_)

    def __PerformDict(self):
        keys = []
        vals = []
        perform = self.__perform
        for i in perform.__dict__.keys():
            if not type(perform).__name__ in i:
                key_ = i
            else:
                key_ = i.split('_' + type(perform).__name__ + '__')[1]
                if key_ in ['a_fpr', 'a_tpr', 'report', 'ft_imp']:
                    continue
                val_ = getattr(perform, key_)
            keys.append(key_)
            vals.append(val_)
        perform_d = dict(zip(keys, vals))
        return perform_d, perform.report

    def TableGen(self, table_name: str = ''):
        perform_d, _ = self.__PerformDict()
        perform_d = [perform_d]
        perform_df = pd.DataFrame(perform_d)
        if not table_name:
            return perform_df
        else:
            save_loc = self.__SaveGen(table_name, 'csv')
            perform_df.to_csv(save_loc, index=False)
            return perform_df

    def FeatImpGen(self, save_name: str = ''):
        series_ = self.__perform.ft_imp
        if not save_name or series_.empty:
            return
        else:
            table_loc = self.__SaveGen(save_name, 'csv')
            chart_loc = self.__SaveGen(save_name, 'png')
            series_.to_csv(table_loc)
            sns.reset_orig()
            ax = sns.barplot(x=series_.values, y=series_.index, errwidth=0.1)
            ax.bar_label(ax.containers[0])
            plt.tight_layout()
            plt.savefig(chart_loc)
            plt.close()

    def FprTprSave(self, save_name: str) -> None:
        df = pd.DataFrame()
        df['fpr'] = self.__perform.a_fpr
        df['tpr'] = self.__perform.a_tpr
        save_loc = self.__SaveGen(save_name, 'csv')
        df.to_csv(save_loc, index=False)

    def TextGen(self, text_name: str) -> None:
        perform_d, report_d = self.__PerformDict()
        save_loc = self.__SaveGen(text_name, 'txt')
        with open(save_loc, 'a') as f:
            f.write(self.__DictToText(perform_d, v_gap=':\t'))
            f.write('\n Report: \n {0}'.format(report_d))
            f.write(':\n\n')

    def RocPlot(self, fig_name: str, fig_dims: tuple = (6, 6)) -> None:
        save_loc = self.__SaveGen(fig_name, 'png')
        plt.subplots(figsize=fig_dims)
        plt.title('Receiver operating characteristic')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.plot(self.__perform.a_fpr,
                 self.__perform.a_tpr,
                 label='ROC AUC = {0}'.format(self.__perform.s_auc))
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_loc)
        plt.close()
