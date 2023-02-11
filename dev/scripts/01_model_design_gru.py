from datetime import datetime

import torch
from loguru import logger

from tentamen.data import datasets
from tentamen.model import Accuracy
from tentamen.settings import presets
from tentamen.train import trainloop

if __name__ == "__main__":
    logger.add(presets.logdir / "01.log")

    trainstreamer, teststreamer = datasets.get_arabic(presets)

    from tentamen.model import GRUmodel
    from tentamen.settings import GRUConfig

    configs = [
        GRUConfig(
            input=13, output=20, tunedir=presets.logdir, hidden_size=64, num_layers=3, dropout=0.4
        ),
    ]

    # 94% ==> hidden_size=16, num_layers=2, dropout=0.3
    # 97,2% hidden_size=32, num_layers=3, dropout=0.3
    # 96,2% hidden_size=64, num_layers=3, dropout=0.3
    # 97% hidden_size=64, num_layers=3, dropout=0.4


    for config in configs:
        model = GRUmodel(config.dict())  # type: ignore

        trainedmodel = trainloop(
            epochs=500,
            model=model,  # type: ignore
            optimizer=torch.optim.Adam,
            learning_rate=1e-3,
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics=[Accuracy()],
            train_dataloader=trainstreamer.stream(),
            test_dataloader=teststreamer.stream(),
            log_dir=presets.logdir,
            train_steps=len(trainstreamer),
            eval_steps=len(teststreamer),
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = presets.modeldir / (timestamp + presets.modelname)
        logger.info(f"save model to {path}")
        torch.save(trainedmodel, path)
