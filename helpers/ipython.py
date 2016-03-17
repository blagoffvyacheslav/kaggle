from data import pd
from IPython.display import display
from IPython.core.pylabtools import figsize
from ipywidgets import FloatProgress


def pd_display(df, max_columns=None):
    old_max_columns = pd.get_option('display.max_columns')
    pd.set_option('display.max_columns', max_columns)
    display(df)
    pd.set_option('display.max_columns', old_max_columns)
