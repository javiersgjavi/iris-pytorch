import os
import torch

from utils.Model import Model
from utils.get_iris import get_iris
from utils.Dataset import LoadDataset
from torch.utils.data import DataLoader


def main():

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = False

    if not os.path.exists('./data/iris.csv'):
        print('No data found. Downloading...')
        get_iris()
        print('Done.')

    dataset = LoadDataset(name='iris', train_size=0.9)
    train_loader = DataLoader(dataset.get_train_data(), batch_size=20, shuffle=True)
    test_loader = DataLoader(dataset.get_test_data())
    input_shape, output_shape = dataset.get_dimensions_models()

    model = Model(input_shape, output_shape, dropout=0.2)
    optimizer = torch.optim.Adam(model.get_parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    model.train(
        train_loader,
        epochs=10,
        optimizer=optimizer,
        loss_func=loss_func,
        device=device
    )

    model.test(test_loader, device=device)


if __name__ == '__main__':
    main()
