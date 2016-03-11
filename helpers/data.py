import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import sklearn as skl
import sklearn.ensemble as skl_ensemble


plt.style.use('ggplot')


class DataStore:
    def __init__(self, store_path='./store/'):
        self.store_path = store_path

    def save(self, data, name, version=1):
        pd.to_pickle(data, self.store_path + 'v{}_{}.pkl'.format(version, name))

    def load(self, name, version=1):
        try:
            return pd.read_pickle(self.store_path + 'v{}_{}.pkl'.format(version, name))
        except IOError:
            return None

    def load_or_create(self, name, loader, version=1):
        data = self.load(name, version)
        if data is None:
            data = loader()
            self.save(data, name, version)

        return data


class DataProvider:
    def __init__(self, files, versions=None, version=None, name='data'):
        self.store = DataStore()
        self.name = name
        self.files = files
        self.dsets = {}
        self.versions = versions if versions is not None else []
        self.load(version)

    def load(self, version):
        if version is None:
            version = len(self.versions)

        dsets = self.store.load(self.name, version)
        if dsets is not None:
            self.dsets = dsets
            return self

        if version == 0:
            self.dsets = {}
            for dset, path in self.files.iteritems():
                self.dsets[dset] = {
                    'data': pd.read_csv(path)
                }
        else:
            self.load(version - 1)
            loader = self.versions[version - 1]
            loader(self)

        self.store.save(self.dsets, self.name, version)
        return self

    def get(self, dset, df='data', col=None):
        data = self.dsets[dset][df]
        if col is not None:
            data = data[col]

        return data

    def extract(self, dset, dfs, encode_categories=True):
        data = []
        for df, cols in dfs:
            data.append(self.get(dset, df=df, col=cols))

        data = pd.concat(data, axis=1)

        if encode_categories:
            category_encode(data)

        return data

    def dset(self, dset):
        return self.dsets[dset]

    def dset_apply_feature(self, dset, name, loader):
        self.dset(dset)[name] = self.store.load_or_create(name + '_' + dset, loader)

    def dfs(self, df='data'):
        return [dset[df] for dset in self.dsets.values()]

    def to_category(self, cols, df='data'):
        for col in cols:
            category_concat(col, self.dfs(df=df))

    def to_datetime(self, cols, df='data'):
        for d in self.dfs(df=df):
            for col in cols:
                d[col] = pd.to_datetime(d[col])


def category_concat(name, dfs):
    dfs = filter(lambda df: name in df, dfs)
    values = pd.concat([df[name] for df in dfs])
    cats = skl.preprocessing.LabelBinarizer().fit(values).classes_
    for df in dfs:
        df[name] = df[name].astype('category', categories=cats)


def category_encode(df):
    for col in df:
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.codes
