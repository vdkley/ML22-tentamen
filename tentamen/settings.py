from pathlib import Path
from typing import Union

from pydantic import BaseModel, HttpUrl
from ray import tune

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float

cwd = Path(__file__)
root = (cwd / "../..").resolve()


class Settings(BaseModel):
    datadir: Path
    testurl: HttpUrl
    trainurl: HttpUrl
    testfile: Path
    trainfile: Path
    modeldir: Path
    logdir: Path
    modelname: str
    batchsize: int
    winningsettings: dict


presets = Settings(
    datadir=root / "data/raw",
    testurl="https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Test_Arabic_Digit.txt",  # noqa N501
    trainurl="https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Train_Arabic_Digit.txt",  # noqa N501
    testfile=Path("ArabicTest.txt"),
    trainfile=Path("ArabicTrain.txt"),
    modeldir=root / "models",
    logdir=root / "logs",
    modelname="model.pt",
    batchsize=128,
    winningsettings = {
        "input" : 13,
        "output" : 20,
        "hidden_size" : 96,
        "num_layers" : 2,
        "dropout": 0.098019
    }
)


class BaseSearchSpace(BaseModel):
    input: int
    output: int
    tunedir: Path

    class Config:
        arbitrary_types_allowed = True


class LinearConfig(BaseSearchSpace):
    h1: int
    h2: int
    dropout: float


class LinearSearchSpace(BaseSearchSpace):
    h1: Union[int, SAMPLE_INT] = tune.randint(16, 128)
    h2: Union[int, SAMPLE_INT] = tune.randint(16, 128)
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.5)


class GRUConfig(BaseSearchSpace):
    hidden_size: int
    num_layers: int
    dropout: float


class GRUSearchSpace(BaseSearchSpace):
    hidden_size: Union[int, SAMPLE_INT] = tune.randint(16, 128)
    num_layers: Union[int, SAMPLE_INT] = tune.randint(2, 5)
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.5)


class AttentionGRUConfig(BaseSearchSpace):
    hidden_size: int
    num_layers: int
    dropout: float

class AttentionGRUSearchSpace(BaseSearchSpace):
    # hidden_size deelbaar door 4 i.v.m. aansluiting op num_heads van de Attention laag
    hidden_size: Union[int, SAMPLE_INT] = tune.choice([i for i in range(80, 116) if i % 4 == 0])
    num_layers: Union[int, SAMPLE_INT] = tune.randint(2, 4)
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.07, 0.18)
