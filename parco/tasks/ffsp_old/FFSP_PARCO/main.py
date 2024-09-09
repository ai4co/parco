import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def


##########################################################################################
# import

import hydra
from omegaconf import DictConfig
from utils.utils import create_logger
from FFSPTrainer import FFSPTrainer as Trainer


##########################################################################################
# main
@hydra.main(version_base="1.3", config_path="../../configs/ffsp", config_name="config.yaml")
def main(cfg: DictConfig):


    env_params = cfg["env"]
    model_params = cfg["model"]
    optimizer_params = cfg["optimizer"]
    trainer_params = cfg["train"]
    tester_params = cfg["test"]
    logger_params = cfg["logger"]
    create_logger(**logger_params)

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params,
                      tester_params=tester_params)

    trainer.run()

    trainer.eval()


##########################################################################################

if __name__ == "__main__":
    main()
