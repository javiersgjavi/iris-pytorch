import os
import pandas as pd
from sklearn import datasets


def get_iris():
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target

    if not os.path.exists('./data'):
        os.mkdir('./data')

    iris_df.to_csv('./data/iris.csv')


if __name__ == '__main__':
    get_iris()
