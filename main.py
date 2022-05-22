import pandas as pd
from itertools import combinations
from sklearn_model import FixReferTest
from Classes.algorithm import RandomForestClassifier, xgb, SVC
from Classes.algorithm import LogisiticReg, KFold, RandomForest


def LogRegSingleTest():
    d_set = {
        'test_per': 0.3,
        'save_name': 'LogReg_med',
        'methods_l': ['4'],
        'targets_l': ['RR', 'V_T', 'VE', 'MPJL_t'],
        'log_para1': {
            'C': 1,
            'max_iter': 2000
        },
        'log_para2': {
            'C': 1,
            'max_iter': 2000,
            'class_weight': 'balanced'
        }
    }
    p_m = FixReferTest()
    p_m.Dataset(d_set['targets_l'], d_set['methods_l'], d_set['test_per'])
    p_m.Modelgen(d_set['log_para1'], LogisiticReg)
    p_m.Resultsav('LogReg', d_set['save_name'])


def LogRegMultiTest():
    result_l = []
    combine = lambda x, y: [i for i in combinations(x, y)]
    d_set = {
        'test_per': 0.3,
        'methods_l': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'targets_l': ['MPJm_t'],
        'log_para1': {
            'C': 1,
            'max_iter': 2000
        },
    }
    for i in range(1, len(d_set['methods_l']) + 1):
        mets_l = combine(d_set['methods_l'], i)
        for mets in mets_l:
            p_m = FixReferTest()
            p_m.Dataset(d_set['targets_l'], list(mets), d_set['test_per'])
            p_m.Modelgen(d_set['log_para1'], LogisiticReg)
            key = '-'.join(mets)
            result_ = {}
            result_['func'] = key
            result_['score'] = round(p_m.result['score'], 2)
            result_['rocauc'] = round(p_m.result['rocauc'], 2)
            result_l.append(result_)
    df = pd.DataFrame(result_l)
    pd.DataFrame.to_csv(df, 'MpMode.csv', index=False)


def KFoldTest():

    pass


def KFoldTest():
    d_set = {
        'test_per': 0.3,
        'fold_name': '5-Fold',
        'file_name': 'RF_5fold_multi',
        # 'methods_l': ['1', '4', '6', '7', '8'],
        # 'targets_l': ['MPJm_t'],
        'methods_l': ['3', '4', '8', '9'],
        'targets_l': ['RR', 'VE', 'V_T', 'MPJm_t'],
        'RF_para': {
            'param_grid': {
                'n_estimators': [700, 1200],
                # 'max_features': ['auto', 'sqrt', 'log2'],
                'max_features': ['log2'],
                'max_depth': [10, 12],
                'criterion': ['entropy']
            },
            'cv': 5,
            'estimator': RandomForestClassifier()
        },
        'SVM_para': {
            'param_grid': {
                'kernel': ['linear'],
                'C': [1.0, 10.0]
            },
            'cv': 5,
            'estimator': SVC()
        }
    }
    p_m = FixReferTest()
    p_m.Dataset(d_set['targets_l'], d_set['methods_l'], d_set['test_per'])
    p_m.Modelgen(d_set['RF_para'], KFold)
    p_m.Resultsav(d_set['fold_name'], d_set['file_name'])


# KFoldTest()
# # LogRegSingleTest()
LogRegSingleTest()
# LogRegMultiTest()