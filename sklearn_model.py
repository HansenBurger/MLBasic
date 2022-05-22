from Classes import balancefunc, processfunc, sourcedata


class Basic():
    def __init__(self) -> None:
        self.__sta = sourcedata.StaticData()
        self.__dyn = sourcedata.DynamicData()

    def DictTextGen(self, dict_):
        func = lambda x: [k + ':\n' + str(v) for k, v in x.items()]
        text = '\n'.join(func(dict_))
        return text

    def __DatasetBuild(self, tar_l, met_l, t_ran):
        process_ = processfunc.PreProcess(self.__sta.label, self.__sta.file)
        process_.FeatureFilt(tar_l, met_l)
        process_.DataSplit(test_size_st=t_ran)
        self.__dyn.d_slc = process_.df
        self.__dyn.d_set = process_.ds

    def __TrainBalance_SMOTE(self):
        dataset = self.__dyn.d_set
        process_ = balancefunc.SMOTE_(dataset['train']['X'],
                                      dataset['train']['y'])
        raw_shape = process_.__repr__()
        process_.OverSample()
        os_shape = process_.__repr__()
        dataset['train']['X'] = process_.X
        dataset['train']['y'] = process_.y
        process_info = '(RAW): {0}\n(OS): {1}'.format(raw_shape, os_shape)
        return process_info

    def __ModelDerivation(self, class_, param):
        dataset = self.__dyn.d_set
        process_ = class_(param)
        process_.Deduce(dataset['train']['X'], dataset['train']['y'])
        process_.Predict(dataset['test']['X'], dataset['test']['y'])
        dataset['result'] = process_.predict_d
        result_ = process_.perform_d
        return result_

    def __ResultGenMain(self, fold_n, file_n, text_list):
        self.__dyn.SaveFoldGen(self.__sta.graph, fold_n)

        true_label = self.__dyn.d_set['test']['y']
        pred_prob = self.__dyn.d_set['result']['prob']

        process_ = processfunc.ResultGen(self.__dyn.s_loc, file_n)
        process_.TextCollect('\n'.join(text_list))
        process_.GraphCollect(true_label, pred_prob)


class FixReferTest(Basic):
    def __init__(self):
        super().__init__()
        self.__text_info = []
        self.__result = {}

    @property
    def result(self):
        return self.__result

    def Dataset(self, tar_l, met_l, t_ran, smote=True):
        self._Basic__DatasetBuild(tar_l, met_l, t_ran)
        if not smote:
            pass
        else:
            os_info = self._Basic__TrainBalance_SMOTE()
            self.__text_info.append(os_info)

    def Modelgen(self, param, func):
        self.__result = self._Basic__ModelDerivation(func, param)
        self.__text_info.append(self.DictTextGen(self.__result))

    def Resultsav(self, fold_n, file_n):
        self._Basic__ResultGenMain(fold_n, file_n, self.__text_info)