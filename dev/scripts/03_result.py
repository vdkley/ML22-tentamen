import torch
from loguru import logger

from tentamen.data import datasets
from tentamen.model import Accuracy
from tentamen.settings import presets
from tentamen.train import evalbatches

if __name__ == "__main__":
    logger.add(presets.logdir / "01.log")

    trainstreamer, teststreamer = datasets.get_arabic(presets)

    # 20230129-140105model.pt

    # timestamp = "20230117-183154"
    timestamp = "20230129-140105"
    path = presets.modeldir / (timestamp + presets.modelname)
    logger.info(f"loading model from {path}")
    model = torch.load(path)

    result = evalbatches(
        model=model,
        testdatastreamer=teststreamer.stream(),
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics=[Accuracy()],
        eval_steps=len(teststreamer),
    )

    logger.info(result)
