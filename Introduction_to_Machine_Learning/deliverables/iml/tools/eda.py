import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from urllib import request
from scipy.io import arff
from io import StringIO
from sklearn.preprocessing import LabelEncoder


def split(df, cat_features=None, response='class'):
    # Split up dataset into features (X) and response variable (y)
    df = df.copy()

    X = df.loc[:, df.columns != response].copy()


    num_features = []
    if cat_features is not None:
        num_features = [cf for cf in list(X.columns) if cf not in cat_features]
    else:
        num_features = [cf for cf in list(X.columns) if cf not in []]

    df[num_features] = df[num_features].astype(float)
    # df[self.response] = df[self.response].astype(str).str.decode("utf-8").copy()

    df[response] = df[response].str.decode("utf-8").copy()
    try:
        df[response] = df[response].astype('int').copy()
        encoder_response = None

    except:
        encoder_response = LabelEncoder()
        encoder_response.fit(df[response])
        df.loc[:, response] = encoder_response.transform(df[response])

    if cat_features is not None:
        df[cat_features] = df[cat_features].stack().str.decode('utf-8').unstack()
        for col in df[cat_features]:
            df[col] = df[col]
        # df[cat_features] = df[cat_features].str.decode("utf-8").copy()

        try:
            df[cat_features] = df[cat_features].astype(int).copy()
        except:
            pass

        X[cat_features] = df[cat_features].copy()


    X[num_features] = df[num_features]
    y = df[response]
    y = pd.DataFrame(y)

    metadata = dict()

    metadata["n_num_features"] = len(X[num_features].columns)
    if cat_features is not None:
        metadata["n_cat_features"] = len(X[cat_features].columns)
    else:
        metadata["n_cat_features"] = 0

    metadata["n_instances"] = len(X)
    metadata["n_features"] = len(X.columns)
    metadata["dtypes"] = df.dtypes.to_dict()

    X = X.copy()

    X_num = X[num_features].copy()
    if cat_features is not None:
        X_cat = X[cat_features].copy()

    else:
        X_cat = None

    splits = {'X': X, 'y': y,
              'X_num': X_num, 'X_cat': X_cat,
              'encoder_response':  encoder_response}

    return splits, metadata


def read_arff(path_data=None, url_data=None):
    """
      Read dataset
    """

    dataset = None
    df = None

    if url_data is not None:
        # Download the dataset
        raw_dataset = request.urlopen(url_data).read().decode('utf8')
        raw_dataset = StringIO(raw_dataset)

        # Parse the dataset into numpy array
        dataset, meta = arff.loadarff(raw_dataset)
    else:
        dataset, meta = arff.loadarff(path_data)

    # Parse the dataset into pandas DataFrame
    df = pd.DataFrame(dataset)

    return df


def check_null(X):
    return X.isnull().sum()


def analyze_num(X_num=None, type_plot='hist'):
    if type_plot == 'hist':
        X_num.plot(kind='hist')
        _ = plt.xlabel("Values")
    elif type_plot == 'box':
        pass


def cat_levels(X_cat):
    cat_value_counts = {}

    for col in X_cat.columns:
        cat_value_counts[col] = X_cat[col].value_counts()

    return cat_value_counts


def analyze_cat(X_cat=None, fig_size=(5, 5)):

    cat_value_counts = {}

    for col in X_cat.columns:
        # f, ax = plt.subplots(figsize=fig_size)
        plt.figure()
        cat_value_counts[col] = X_cat[col].value_counts()
        _ = sns.barplot(cat_value_counts[col].index,
                        cat_value_counts[col].values, alpha=0.9)
        # plt.bar(cat_value_counts[col].index, cat_value_counts[col].values, alpha=0.9)

        plt.title(f'Frequency: {col}')
        plt.ylabel('# Instances')
        # time.sleep(0.5)

        plt.show()
        # plt.pause(0.05)

        t = cat_value_counts[col]
        d = {'value': t.index,
             'freq': t.values,
             '%': np.round(t.values / t.values.sum(), 2)}
        print(pd.DataFrame(d).to_string(index=False))
        print('-' * 30)

        
def cum_variance(variances):
    cum_var_exp = np.cumsum(variances)*100

    # Compute the cumulative explained variance
    x = np.linspace(0,len(variances)-1,len(variances)-1)
    y1 = [90 for i in range(len(variances)-1)]
    y2 = [80 for i in range(len(variances)-1)]
    y3 = [60 for i in range(len(variances)-1)]
    plt.plot(cum_var_exp,label='Cumulative explained variance')
    plt.plot(x,y1,linestyle = ':', color='b',label='90% Explained variance')
    plt.plot(x,y2,linestyle = ':', color='g',label='80% Explained variance')
    plt.plot(x,y3,linestyle = ':', color='y',label='60% Explained variance')
    plt.title('Cumulative explained variance as a function of the num of components')
    plt.legend(loc='lower right')
    plt.show()