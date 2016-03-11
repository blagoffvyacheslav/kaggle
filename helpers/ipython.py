import pandas as pd
from IPython.display import display


def pd_display_all(df):
    old = pd.get_option('display.max_columns')
    pd.set_option('display.max_columns', None)
    display(df)
    pd.set_option('display.max_columns', old)
