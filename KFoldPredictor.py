from pathlib import Path
from pandas import DataFrame
from Classes.processing import DataSetProcess
from Classes.balancing import BalanSMOTE, BalanRandOS
from Classes.algorithm import ParaSel_Grid, ParaSel_Rand


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

    def __GetEvalSet(self, data_set):
        eval_set = [(data_set['X_train'], data_set['y_train']),
                    (data_set['X_test'], data_set['y_test'])]
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
                      re_select_feat: bool = False):
        '''
        Cross-validation
        para_init_add: Model init s_param
        para_deduce_add: Model deduce param
        re_select_feat
        '''

        for fold in range(self.__fold_num):
            para_init = self.__fold_para[fold]
            para_init.update(para_init_add)
            main_p = self.__algo_type(self.__fold_data[fold], para_init)

            if not re_select_feat:
                para_deduce = {}
                para_deduce.update(para_deduce_add)
                main_p.Deduce(para_deduce)
                self.__pred_rsts.append(main_p.Predict())
            else:
                para_deduce = {
                    'eval_set': self.__GetEvalSet(main_p.dataset),
                    'early_stopping_rounds': 10,
                    'eval_metric': 'auc',
                    'verbose': True
                }
                para_deduce.update(para_deduce_add)
                main_p.Deduce(para_deduce)
                _ = main_p.GetFeatureImportance()

                para_deduce['eval_set'] = self.__GetEvalSet(main_p.dataset)
                para_deduce['early_stopping_rounds'] = 50
                main_p.Deduce(para_deduce)
                self.__pred_rsts.append(main_p.Predict())

    def ResultGenerate(self, save_path: Path):
        '''

        '''
        repr_gen = lambda dict_: ('\n').join(k + ':\t' + str(v)
                                             for k, v in dict_.items())

        ave_auc = sum([fold['rocauc']
                       for fold in self.__pred_rsts]) / self.__fold_num
        ave_r2 = sum([fold['score']
                      for fold in self.__pred_rsts]) / self.__fold_num

        with open(save_path / 'pred_result.txt', 'w') as f:
            for i in range(self.__fold_num):
                fold_info = self.__pred_rsts[i]
                fold_para = self.__para_rsts[i]

                f.write('\n{0}-Fold:\n'.format(i))
                f.write('SCORE: \t {0} \n'.format(fold_info['score']))
                f.write('ROCAUC: \t {0} \n'.format(fold_info['rocauc']))
                f.write('REPORT: \n {0} \n'.format(fold_info['report']))
                f.write('PARAMS: \n {0} \n'.format(repr_gen(fold_para)))

            f.write('\nAVE Performance:\n')
            f.write('SCORE:\t{0}\n'.format(ave_r2))
            f.write('ROCAUC:\t{0}\n'.format(ave_auc))

        for i in range(self.__fold_num):
            fold_info = self.__pred_rsts[i]
            fold_data = self.__data_sets[i]
            save_name = '{0}-Fold_ROC.png'.format(i)
            # main_p = PlotMain(save_path)
            # main_p.RocSinglePlot(fold_data[3], fold_info['prob'], save_name)

        pred_df = DataFrame()
        pred_df['mode'] = [
            'fold_' + str(i + 1) for i in range(self.__fold_num)
        ] + ['ave']
        pred_df['auc'] = [round(i['rocauc'], 2)
                          for i in self.__fold_pred] + [round(ave_auc, 2)]
        pred_df['r2'] = [round(i['score'], 2)
                         for i in self.__fold_pred] + [round(ave_r2, 2)]

        pred_df.set_index('mode', drop=True)
        DataFrame.to_csv(pred_df, save_path / 'pred_result.csv', index=False)
