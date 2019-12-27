import yaml

from texar.torch import HParams
from forte.train_pipeline import TrainPipeline
from forte.trainer.dense_trainer import DenseTrainer

from reader import TrainReader


def main():
    config = yaml.safe_load(open("train_config.yml", "r"))

    config = HParams(config, default_hparams=None)

    reader = TrainReader()

    trainer = DenseTrainer()

    train_pipe = TrainPipeline(train_reader=reader, trainer=trainer,
                               dev_reader=reader, configs=config)
    train_pipe.run()


if __name__ == '__main__':
    main()
