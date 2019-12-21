import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tools.eda import cat_levels
from sklearn.decomposition import PCA


def scale(X_num):
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num.values)
    X_num_scaled = pd.DataFrame(X_num_scaled,
                                index=X_num.index, columns=X_num.columns)

    return X_num_scaled


def encode(X_cat):
    binary_cat_features = []
    cat_value_counts = cat_levels(X_cat)

    for k, v in cat_value_counts.items():
#         print(f'Feature: {k:20} | # categories: {v.count()}')
        if v.count() == 2:
            binary_cat_features.append(k)

    # print()
    # print(f'Binary cat_features: {binary_cat_features}')
    # rest_cat_features = [cf for cf in list(X_cat.columns) if cf not in binary_cat_features]
    rest_cat_features = [cf for cf in list(X_cat.columns)]
    # rest_cat_features = list(set(X_cat.columns)-set(binary_cat_features))
    # print(f'Remaining cat_features: {rest_cat_features}')

    # One hot encoding

    X_cat_encoded = pd.get_dummies(X_cat, columns=rest_cat_features)

    return X_cat_encoded


def join_features(X_num, X_cat):
    X_total = pd.concat([X_num, X_cat], axis=1)
    # print(f'# Total features: {len(X_total.columns)}')

    return X_total


def graph_components(X, n_components=0.9):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    list_components = list(range(pca.n_components_))
    plt.bar(list_components, pca.explained_variance_ratio_)
    plt.xlabel('Components')
    plt.ylabel('Variance %')
    plt.xticks(list_components)

    # X_pca = pd.DataFrame(X_pca)
    

def binning(X, n_classes=3):
    X = X.copy()
    for col in X.columns:
        X[col] = pd.cut(X[col], bins=n_classes, labels=range(n_classes))
        X[col] = X[col].astype(int)
        # X[col] = pd.qcut(X[col], q=n_classes, labels=range(n_classes))

    return X
