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

def clean_sick2(df):

    """
        Clean mixed data: cmc.
    """

    cat_features = ['sex', 'on_thyroxine',
                    'query_on_thyroxine', 'on_antithyroid_medication',
                    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
                    'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                    'goitre', 'tumor', 'hypopituitary', 'psych',
                    'TSH_measured', 'T3_measured', 'TT4_measured',
                    'T4U_measured', 'FTI_measured', 'TBG_measured',
                    'referral_source'] # for sick dataset
    response = 'Class' # for sick dataset
    splits, metadata = eda.split(df, cat_features=cat_features,
                                 response=response)
    X_num = splits['X_num']
    X_cat = splits['X_cat']

    # Drop columns with many nan
    X_num.drop(['TBG'], axis=1, inplace=True)
    X_num = X_num.fillna(X_num.mean())
    # Outliers
    # print(f'# Samples before removing outliers: {len(X_num)}')
    rows_to_remove = (np.abs(stats.zscore(X_num)) < 3).all(axis=1)
    X_num = X_num[rows_to_remove].copy()
    # print(f'# Samples after removing outliers: {len(X_num)}')
    # y = splits['y'][response].values
    y = splits['y'][rows_to_remove][response].values

    # Scaling
    X_num_scaled = prep.scale(X_num)

    # Removing categ. levels
    # X_cat = X_cat[rows_to_remove].copy()
    pd.options.mode.chained_assignment = None

    # X_cat.loc[X_cat['heducation'] == 1, 'heducation'] = 2
    # X_cat.loc[X_cat['hoccupation'] == 4, 'hoccupation'] = 3

    # Encoding
    X_cat_encoded = prep.encode(X_cat)

    return X_num_scaled, X_cat_encoded, y

def clean_sick(df, encoder=None):

    """
        Clean mixed data: cmc.
    """

    cat_features = ['sex', 'on_thyroxine',
                    'query_on_thyroxine', 'on_antithyroid_medication',
                    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
                    'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                    'goitre', 'tumor', 'hypopituitary', 'psych',
                    'TSH_measured', 'T3_measured', 'TT4_measured',
                    'T4U_measured', 'FTI_measured', 'TBG_measured',
                    'referral_source'] # for sick dataset
    response = 'Class' # for sick dataset





    splits, metadata = eda.split(df, cat_features=cat_features,
                                 response=response)
    X_num = splits['X_num']
    X_num.drop(['TBG'], axis=1, inplace=True)
    #print(X_num)

    X_cat = splits['X_cat']
    #print(X_cat)

    y = splits['y'][response].values

    # Drop columns with many nan

    # Replace values by the median of the column
    from sklearn.impute import SimpleImputer
    imp_mean = SimpleImputer( strategy='median') #for median imputation replace 'mean' with 'median'
    imp_mean.fit(X_num)
    X_num_no_nan = imp_mean.transform(X_num)
    #print(X_num_no_nan)

    # The data set is converted to data frame again
    X_num_ok = pd.DataFrame(X_num_no_nan, columns=X_num.columns)

    # Scaling
    X_num_scaled = (X_num_ok - X_num_ok.min()) / (X_num_ok.max() - X_num_ok.min())


    pd.options.mode.chained_assignment = None

    # Encoding
    X_cat_encoded, encoder = prep.encode(X_cat, encoder)

    return pd.DataFrame(X_num_scaled), pd.DataFrame(X_cat_encoded), pd.DataFrame(y), encoder
