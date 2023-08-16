import argparse

import yaml

from aml.train import PotentialTrainer

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

trainer = PotentialTrainer.from_config(config)
trainer.train()
