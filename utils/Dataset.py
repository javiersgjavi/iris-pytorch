import torch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, TensorDataset


class LoadDataset(Dataset):
    def __init__(self, name, train_size):
        self.data = pd.read_csv(f'./data/{name}.csv')

        x = self.data.iloc[:, :-1]
        y = self.data['target']

        x_norm = (x - x.mean()) / x.std()

        separator = int(train_size * len(x))
        ohe = OneHotEncoder()
        ohe.fit(y.values.reshape(-1, 1))

        y_train = pd.DataFrame(ohe.transform(y[:separator].values.reshape(-1, 1)).toarray())
        y_test = pd.DataFrame(ohe.transform(y[separator:].values.reshape(-1, 1)).toarray())

        x_train, x_test = x_norm[:separator], x_norm[separator:]

        self.x_train = torch.tensor(x_train.values, dtype=torch.float)
        self.y_train = torch.tensor(y_train.values, dtype=torch.float)
        self.x_test = torch.tensor(x_test.values, dtype=torch.float)
        self.y_test = torch.tensor(y_test.values, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

    def get_train_data(self):
        return TensorDataset(self.x_train, self.y_train)

    def get_test_data(self):
        return TensorDataset(self.x_test, self.y_test)

    def get_dimensions_models(self):
        return self.x_train.shape[1], self.y_train.shape[1]
