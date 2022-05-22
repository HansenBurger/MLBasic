import sys
from peewee import *
from pathlib import Path
from playhouse.sqlite_ext import JSONField

sys.path.append(str(Path.cwd()))

db = SqliteDatabase(r'C:\\Main\\Data\\_\\Database\\sqlite\\RespData_2205.db')


class ZresParam(Model):

    index = AutoField()
    pid = IntegerField(column_name='patient_id')
    rid = TextField(column_name='record_id')
    rec_t = DateTimeField(column_name='record_time')
    rec_i = IntegerField(column_name='data_index')
    rec_f = IntegerField(column_name='record_flag')

    class Meta:
        table_name = 'zres_param'
        database = db


class OutcomeExWean(Model):

    pid = IntegerField(primary_key=True)
    icu = TextField()
    ex_t = DateTimeField(column_name='ExtubeTime')
    ex_s = TextField(column_name='ExtubeStatus')
    we_t = DateTimeField(column_name='WeanTime')
    we_s = TextField(column_name='WeanStatus')

    class Meta:
        table_name = 'Outcome_ExtubeWean'
        database = db


class ExtubePrep(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    tail_t = DateTimeField(column_name='TAIL_t', null=True)
    rot = TextField(column_name='route', null=True)
    rec_t = DateTimeField(column_name='REC_t', null=True)
    zdt = TextField(column_name='ZDT', null=True)
    zpx = TextField(column_name='ZPX', null=True)
    opt = BooleanField(column_name='op_tag', null=True)
    v_t = IntegerField(column_name='vent_time', null=True)
    mch = TextField(column_name='machine', null=True)
    vmd = JSONField(column_name='vent_mode', null=True)
    spd = JSONField(column_name='peep_ps', null=True)

    class Meta:
        table_name = 'Extube_Prep'
        database = db

    @classmethod
    def get_as_dict(cls, expr):
        query = cls.select().where(expr).dicts()
        return query.get()

    def ObjToDict(self):
        dict_ = {
            'pid': self.pid,
            'icu': self.icu,
            'e_t': self.e_t,
            'e_s': self.e_s,
            'rid': self.rid,
            'tail_t': self.tail_t,
            'rot': self.rot,
            'rec_t': self.rec_t,
            'zdt': self.zdt,
            'zpx': self.zpx,
            'opt': self.opt,
            'v_t': self.v_t,
            'mch': self.mch,
            'vmd': self.vmd,
            'spd': self.spd
        }
        return dict_


class WeanPrep(Model):

    index = AutoField()
    pid = IntegerField(column_name='PID')
    icu = TextField(column_name='ICU')
    e_t = DateTimeField(column_name='END_t')
    e_s = TextField(column_name='END_s')
    rid = TextField(column_name='RID')
    tail_t = DateTimeField(column_name='TAIL_t', null=True)
    rot = TextField(column_name='route', null=True)
    rec_t = DateTimeField(column_name='REC_t', null=True)
    zdt = TextField(column_name='ZDT', null=True)
    zpx = TextField(column_name='ZPX', null=True)
    opt = BooleanField(column_name='op_tag', null=True)
    v_t = IntegerField(column_name='vent_time', null=True)
    mch = TextField(column_name='machine', null=True)
    vmd = JSONField(column_name='vent_mode', null=True)
    spd = JSONField(column_name='peep_ps', null=True)

    class Meta:
        table_name = 'Wean_Prep'
        database = db

    def ObjToDict(self):
        dict_ = {
            'pid': self.pid,
            'icu': self.icu,
            'e_t': self.e_t,
            'e_s': self.e_s,
            'rid': self.rid,
            'tail_t': self.tail_t,
            'rot': self.rot,
            'rec_t': self.rec_t,
            'zdt': self.zdt,
            'zpx': self.zpx,
            'opt': self.opt,
            'v_t': self.v_t,
            'mch': self.mch,
            'vmd': self.vmd,
            'spd': self.spd
        }
        return dict_