import sys
sys.path.append('..')
from helpers.data import *


def data_v1(data):
    for df in data.dfs().itervalues():
        df.index = df['ID']
        df.drop(['ID'], axis=1, inplace=True)


def data_v2(data):
    cat_types = list(data.get('train').dtypes[data.get('train').dtypes == 'O'].index)
    for df in data.dfs().itervalues():
        df[cat_types] = df[cat_types].fillna('UNKNOWN')

    data.to_category(cat_types)


def data_v3(data):
    for dset, df in data.dfs().iteritems():
        nans = pd.DataFrame(index=df.index)
        nans['nan_count'] = df.apply(lambda row: row.isnull().sum(), axis=1)
        data.dset(dset)['nans'] = nans

data = DataProvider({
    'train': 'data/train.csv',
    'test': 'data/test.csv'
}, [data_v1, data_v2, data_v3])
