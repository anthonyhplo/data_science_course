import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


if __name__ == '__main__':
    df_train = pd.read_csv("../dataset/mnist/train.csv")

    X = df_train.iloc[:,1:]
    y = df_train.iloc[:,0]

    pca = PCA()
    X_transformed = pca.fit_transform(X)

    plt.scatter(X_transformed[:,0], X_transformed[:,1], c=y, alpha=0.5)
    plt.show()

    cumulative = []
    last = 0
    for v in pca.explained_variance_ratio_:
        cumulative.append(last + v)
        last = cumulative[-1]

    plt.plot(cumulative)
    plt.show()