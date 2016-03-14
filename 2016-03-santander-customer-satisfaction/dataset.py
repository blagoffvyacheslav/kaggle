import sys
sys.path.append('..')
from helpers.data import *
import itertools


def data_v1(data):
    for df in data.dfs().itervalues():
        df.index = df['ID']
        df.drop(['ID'], axis=1, inplace=True)


def data_v2(data):
    describe = data.get('train').describe().T
    cols = describe[describe['std'] == 0].index
    for df in data.dfs().itervalues():
        df.drop(cols, axis=1, inplace=True)


def data_v3(data):
    duplicates = {}

    def add_duplicate(col1, col2):
        if col1 in duplicates:
            duplicates[col1].append(col2)
        else:
            duplicates[col1] = [col2]

    train = data.get('train')
    for col1, col2 in itertools.combinations(train.columns, 2):
        if train[col1].equals(train[col2]):
            add_duplicate(col1, col2)

    for df in data.dfs().itervalues():
        for col, cols in duplicates.iteritems():
            if col in df:
                dcols = [col] + cols
                df['+'.join(dcols)] = df[col]
                df.drop(dcols, axis=1, inplace=True)


data = DataProvider({
    'train': 'data/train.csv',
    'test': 'data/test.csv'
}, [data_v1, data_v2, data_v3])
