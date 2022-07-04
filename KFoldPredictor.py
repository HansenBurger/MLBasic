import sys
from pathlib import Path
from numpy import mean
from pandas import concat
from pandas import DataFrame

from MLBasic.Classes.processing import DataSetProcess, OutcomeGenerate
from MLBasic.Classes.balancing import BalanSMOTE, BalanRandOS
from MLBasic.Classes.algorithm import ParaSel_Grid, ParaSel_Rand


class Basic():
    def __init__(self):
        pass


class KFoldMain(Basic):
    def __init__(self, algo_class: any, split_n: int = 5) -> None:
        super().__init__()
        self.__algo_type = algo_class
        self.__fold_num = split_n
        self.__data_sets = []
        self.__para_slts = []
        self.__pred_rsts = []
        self.__feat_imps = []

    def __GetEvalSet(self, data_set):
        eval_set = [(data_set.X_train, data_set.y_train),
                    (data_set.X_test, data_set.y_test)]
        return eval_set

    def DataSetBuild(self, data_: DataFrame, col_label: str):
        '''
        Data reading, updating, balancing
        data_: feature data
        col_label: colname of label
        '''
        main_p = DataSetProcess(data_, col_label)
        main_p.DataImpute(impute_type='knn')
        data_l = main_p.KFoldSplit(self.__fold_num)

        for data_ in data_l:
            try:
                balance_p = BalanSMOTE(data_[0], data_[1])
                data_[0], data_[1] = balance_p.OverSample()
            except:
                balance_p = BalanRandOS(data_[0], data_[1])
                data_[0], data_[1] = balance_p.OverSample()

            self.__data_sets.append(data_)

    def ParamSelectRand(self, para_pool: dict, eval_set: bool = False):
        '''
        Select best param for training using RandomizedSearchCV
        para_pool: super params for filting
        eval_set: Use early stop or not
        '''
        para_init = {
            'estimator': self.__algo_type().algorithm(),
            'param_distributions': para_pool,
            'scoring': 'accuracy',
            'cv': 3,
        }

        for i in range(self.__fold_num):
            main_p = ParaSel_Rand(self.__data_sets[i], para_init)
            para_deduce = {
                'eval_set': self.__GetEvalSet(main_p.dataset),
                'early_stopping_rounds': 10,
                'verbose': True
            } if eval_set else {}
            main_p.Deduce(para_deduce)
            self.__para_slts.append(main_p.BestParam())

    def CrossValidate(self,
                      para_init_add: dict = {},
                      para_deduce_add: dict = {},
                      re_select_feat: bool = False,
                      get_feat_imp: bool = False):
        '''
        Cross-validation
        para_init_add: Model init s_param
        para_deduce_add: Model deduce param
        re_select_feat
        '''

        for fold in range(self.__fold_num):
            para_init = self.__para_slts[fold]
            para_init.update(para_init_add)
            main_p = self.__algo_type(self.__data_sets[fold], para_init)

            if not re_select_feat:
                para_deduce = {}
                para_deduce.update(para_deduce_add)
            else:
                para_deduce = {
                    'eval_set': self.__GetEvalSet(main_p.dataset),
                    'early_stopping_rounds': 10,
                    'eval_metric': 'auc',
                    'verbose': True
                }
                para_deduce.update(para_deduce_add)
                main_p.Deduce(para_deduce)
                main_p.RebootByImpFeats()
                para_deduce['eval_set'] = self.__GetEvalSet(main_p.dataset)
                para_deduce['early_stopping_rounds'] = 50

            main_p.Deduce(para_deduce)
            main_p.Predict()
            if get_feat_imp:
                main_p.GetFeatureImp()
            self.__pred_rsts.append(main_p.perform)

    def ResultGenerate(self,
                       store_results: bool = True,
                       save_path: Path = Path.cwd()):
        '''
        '''
        df_tot = []
        if store_results:
            save_name = 'pred_result'
            save_path.mkdir(parents=True, exist_ok=True)
        for i in range(self.__fold_num):
            main_p = OutcomeGenerate(self.__pred_rsts[i], save_path)
            if store_results:
                main_p.TextGen(save_name)  # write into same file
                main_p.RocPlot('{0}-Fold_ROC'.format(i + 1))
                main_p.FeatImpGen('{0}-Fold_Imp'.format(i + 1))
            df_fold = main_p.TableGen()
            df_fold['mode'] = 'fold_' + str(i + 1)
            df_tot.append(df_fold)
        df_tot = concat(df_tot, axis=0, ignore_index=True)

        ave_keys = [c for c in df_tot.columns if c != 'mode']
        ave_values = [round(mean(df_tot[k].tolist()), 3) for k in ave_keys]
        ave_dict = dict(zip(ave_keys, ave_values))
        ave_dict['mode'] = 'ave'

        df_tot = df_tot.append(ave_dict, ignore_index=True)
        df_tot = df_tot.set_index('mode', drop=True)

        if store_results:
            df_tot.to_csv(save_path / (save_name + '.csv'))

        return df_tot