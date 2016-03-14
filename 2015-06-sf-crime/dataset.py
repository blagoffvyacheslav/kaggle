import sys
sys.path.append('..')
from helpers.data import *


def data_v1(data):
    data.to_datetime(['Dates'])
    data.to_category(['DayOfWeek', 'PdDistrict', 'Address', 'Category', 'Descript', 'Resolution'])


def data_v2(data):
    for dset, df in data.dfs().iteritems():
        data.dset(dset)['dates'] = datetime_decompose_series(df['Dates'])


def data_v3(data):
    def get_streets(street):
        streets = map(str.strip, street.split('/'))
        if len(streets) == 1:
            streets[0] = re.sub('.* Block of ', '', streets[0])

        return streets

    for dset, df in data.dfs().iteritems():
        streets = df['Address'].apply(get_streets)
        addr = data.dset(dset)['addr'] = pd.DataFrame(index=df.index)
        addr['street'] = streets.apply(lambda s: '_'.join(s))
        addr['corner'] = streets.apply(lambda s: len(s) > 1)

    data.to_category(['street'], df='addr')


data = DataProvider({
    'train': 'data/train.csv',
    'test': 'data/test.csv'
}, [data_v1, data_v2, data_v3])
