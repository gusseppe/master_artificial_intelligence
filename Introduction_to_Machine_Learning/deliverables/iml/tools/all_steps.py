import pandas as pd
import numpy as np

from scipy import stats
from tools import eda
from tools import preprocess as prep


def clean_cmc(df):

    """
        Clean mixed data: cmc.
    """

    cat_features = ['weducation', 'heducation', 'wreligion',
                    'wworking', 'hoccupation', 'living_index', 'media_exposure']

    splits, metadata = eda.split(df, cat_features=cat_features,
                                 response='class')
    X_num = splits['X_num']
    X_cat = splits['X_cat']

    # Outliers
    print(f'# Samples before removing outliers: {len(X_num)}')
    rows_to_remove = (np.abs(stats.zscore(X_num)) < 3).all(axis=1)
    X_num = X_num[rows_to_remove].copy()
    print(f'# Samples after removing outliers: {len(X_num)}')
    y = splits['y'][rows_to_remove]['class'].values

    # Scaling
    X_num_scaled = prep.scale(X_num)

    # Removing categ. levels
    X_cat = X_cat[rows_to_remove].copy()
    pd.options.mode.chained_assignment = None

    X_cat.loc[X_cat['heducation'] == 1, 'heducation'] = 2
    X_cat.loc[X_cat['hoccupation'] == 4, 'hoccupation'] = 3

    # Encoding
    X_cat_encoded = prep.encode(X_cat)

    return X_num_scaled, X_cat_encoded, y
