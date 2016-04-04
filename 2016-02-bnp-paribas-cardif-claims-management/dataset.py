from helpers.data import *
from helpers.model import LogisticXGB, model_train_cv_parallel
from sklearn.metrics import log_loss


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


def data_v5(data):
    def get_y():
        return data.get('train', 'y')

    def get_X(dset='train'):
        return data.extract(dset, [
            ('nums', None),
            ('cats', None),
            ('nans', None),
        ]).fillna(-999)

    # 0.46394
    model = LogisticXGB(n_estimators=350, learning_rate=0.05, max_depth=7, seed=42)
    X, y = get_X(), get_y()
    X_test = get_X('test')

    cv = skl.cross_validation.StratifiedKFold(y, n_folds=16, shuffle=True, random_state=1234)
    data.dset('train')['models'] = pd.DataFrame(index=X.index)
    data.get('train', 'models')['draft_xgb'] = model_train_cv_parallel(model, X, y, n_jobs=1, cv=cv)['predict'] - 0.5

    model.fit(X, y)
    data.dset('test')['models'] = pd.DataFrame(index=X_test.index)
    data.get('test', 'models')['draft_xgb'] = model.predict_proba(X_test)[:, 1] - 0.5


def data_v6(data):
    def get_y():
        return data.get('train', 'y')

    def get_X(dset='train'):
        return data.extract(dset, [
            ('nums', None),
            ('cats', None),
            ('nans', None),
        ]).fillna(-999)

    # 0.46007
    model = skl.ensemble.ExtraTreesClassifier(n_estimators=1000, criterion='entropy', min_samples_leaf=5, max_features=0.8, n_jobs=8, random_state=42)
    X, y = get_X(), get_y()
    X_test = get_X('test')

    cv = skl.cross_validation.StratifiedKFold(y, n_folds=16, shuffle=True, random_state=1234)
    data.get('train', 'models')['draft_ext'] = model_train_cv_parallel(model, X, y, n_jobs=1, cv=cv)['predict'] - 0.5

    model.fit(X, y)
    data.get('test', 'models')['draft_ext'] = model.predict_proba(X_test)[:, 1] - 0.5


data = DataProvider({
    'train': 'data/train.csv',
    'test': 'data/test.csv'
}, [data_v1, data_v2, data_v3, data_v4, data_v5, data_v6])


def kaggle_split(X, y):
    return skl.cross_validation.train_test_split(X, y, train_size=0.7, random_state=38)
