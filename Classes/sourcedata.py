from pathlib import Path
from datetime import datetime


class Basic():
    def __init__(self) -> None:
        pass

    def TimeNow(self):
        now = datetime.now()
        fold_pre = '{0}{1}{2}_{3}_'.format(now.year,
                                           str(now.month).rjust(2, '0'),
                                           str(now.day).rjust(2, '0'),
                                           str(now.hour).rjust(2, '0'))
        return fold_pre


class StaticData(Basic):
    def __init__(self) -> None:
        self.__graph = r'C:\Main\Data\_\Result\Graph'
        self.__form = r'C:\Main\Data\_\Result\Form'
        self.__file = r'source\data.csv'
        self.__label = 'endo_end'

    @property
    def label(self):
        return self.__label

    @property
    def file(self):
        return self.__file

    @property
    def graph(self):
        return self.__graph

    @property
    def form(self):
        return self.__form


class DynamicData(Basic):
    def __init__(self):
        super().__init__()
        self.__d_slc = None
        self.__d_set = None
        self.__s_loc = None

    def SaveFoldGen(self, parent, module_name=None):
        folder_name = self.TimeNow() + module_name
        save_loc = Path(parent) / folder_name
        save_loc.mkdir(parents=True, exist_ok=True)
        self.__s_loc = save_loc

    @property
    def s_loc(self):
        return self.__s_loc

    @property
    def d_slc(self):
        return self.__d_slc

    @d_slc.setter
    def d_slc(self, v):
        self.__d_slc = v

    @property
    def d_set(self):
        return self.__d_set

    @d_set.setter
    def d_set(self, v):
        self.__d_set = v
