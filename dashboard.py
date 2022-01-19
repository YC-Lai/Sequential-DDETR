import os
from pathlib import Path
from comet_ml import Experiment, ExistingExperiment


class Dashboard:
    """Record training/evaluation statistics to comet
    """

    def __init__(self, args, workspace: str, api_key: str):
        self.experiment = Experiment(api_key=api_key,
                                     project_name="Sequential-ddtr",
                                     workspace=workspace)
        self.experiment.log_parameters(vars(args))
        
        self.epochs = 1
        self.global_step = 1
        self.status = "evaluating"

    def set_status(self, status: str):
        ## training/ evaluating
        assert status == "training" or status == "evaluating"
        self.status = status

    def step(self):
        self.global_step += 1
        
    def update_epoch(self):
        self.epochs += 1

    def log_metrics(self, metrics: dict):
        if self.status == "training":
            with self.experiment.train():
                self.experiment.log_metrics(metrics, step=self.global_step)
        else:
            with self.experiment.validate():
                self.experiment.log_metrics(metrics, step=self.epochs)

    def check(self):
        if not self.experiment.alive:
            print("Comet logging stopped")
