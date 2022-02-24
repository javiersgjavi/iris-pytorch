import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(MLP, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(300, output_dim),
        )

    def forward(self, x):
        logits = self.model(x)
        return self.softmax(logits)


class Model:
    def __init__(self, input_dim, output_dim, dropout):
        self.model = MLP(input_dim=input_dim, output_dim=output_dim, dropout=dropout)

    def train(self, train_loader, optimizer, loss_func, epochs, device=False):
        if device:
            self.model.to(device)

        for epoch in range(epochs):
            for batch, (x, y) in enumerate(train_loader):
                if device:
                    x = x.to(device)
                    y = y.to(device)

                pred = self.model(x)
                loss = loss_func(y, pred)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss, current = loss.item(), batch * len(x)
            print(f'Epoch: {epoch + 1}/{epochs} | Loss: {loss:.4f}')

    def test(self, data, device=False):
        correct = 0
        with torch.no_grad():
            for x, y in data:
                if device:
                    x = x.to(device)
                    y = y.to(device)

                pred = self.model(x)
                correct += (pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()

        accuracy = correct / len(data)
        print(f'Accuracy: {round(accuracy * 100, 2)}%')

    def get_parameters(self):
        return self.model.parameters()
