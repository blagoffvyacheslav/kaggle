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


def data_v4(data):
    train = data.get('train')
    data.dset('train')['y'] = train['target']

    train = train.drop(['target'], axis=1)
    cat_types = list(train.select_dtypes(include=['category']).columns)
    num_types = list(train.select_dtypes(exclude=['category']).columns)

    for dset, df in data.dfs().iteritems():
        data.dset(dset)['cats'] = df[cat_types]
        data.dset(dset)['nums'] = df[num_types]

data = DataProvider({
    'train': 'data/train.csv',
    'test': 'data/test.csv'
}, [data_v1, data_v2, data_v3, data_v4])


def kaggle_split(X, y):
    return skl.cross_validation.train_test_split(X, y, train_size=0.7, random_state=38)
