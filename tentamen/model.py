from typing import Callable, Dict, Protocol

import torch
import torch.nn as nn

Tensor = torch.Tensor


class GenericModel(Protocol):
    train: Callable
    eval: Callable
    parameters: Callable

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass


class Linear(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        #self.conv = nn.Conv1d(128, config["input"], kernel_size, stride=2)
        self.encoder = nn.Sequential(
            nn.Linear(config["input"], config["h1"]),
            nn.ReLU(),
            nn.Linear(config["h1"], config["h2"]),
            nn.Dropout(config["dropout"]),
            nn.ReLU(),
            nn.Linear(config["h2"], config["output"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('before mean:')
        # print(x.shape)
        #x = x[:, -1, :]
        x = x.mean(dim=1)
        #x = x.reshape(128, 13)
        # print('after mean:')
        # print(x.shape)
        x = self.encoder(x)
        return x


class GRUmodel(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config["input"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.linear = nn.Linear(config["hidden_size"], config["output"])

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class Accuracy:
    def __repr__(self) -> str:
        return "Accuracy"

    def __call__(self, y: Tensor, yhat: Tensor) -> Tensor:
        """
        yhat is expected to be a vector with d dimensions.
        The highest values in the vector corresponds with
        the correct class.
        """
        return (yhat.argmax(dim=1) == y).sum() / len(yhat)



class DimensionReducer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, height, width) -> (batch_size, width, height)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (batch_size, width, height // stride) -> (batch_size, height // stride, width)
        return x

