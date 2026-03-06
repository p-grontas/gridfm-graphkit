import os
from abc import ABC, abstractmethod
import lightning as L
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import Mapping


class BaseTask(L.LightningModule, ABC):
    """
    Abstract base LightningModule for feature reconstruction tasks.
    Contains shared training/validation/test logic, logging, and optimizer setup.
    """

    def __init__(self, args, data_normalizers):
        super().__init__()
        self.args = args
        self.data_normalizers = data_normalizers
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass"""
        pass

    @abstractmethod
    def training_step(self, batch):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    @abstractmethod
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    @rank_zero_only
    def on_fit_start(self):
        # Save normalization statistics
        if isinstance(self.logger, MLFlowLogger):
            log_dir = os.path.join(
                self.logger.save_dir,
                self.logger.experiment_id,
                self.logger.run_id,
                "artifacts",
                "stats",
            )
        else:
            log_dir = os.path.join(self.logger.save_dir, "stats")

        os.makedirs(log_dir, exist_ok=True)

        # Human-readable log
        log_stats_path = os.path.join(log_dir, "normalization_stats.txt")
        with open(log_stats_path, "w") as log_file:
            for i, normalizer in enumerate(self.data_normalizers):
                log_file.write(
                    f"Data Normalizer {self.args.data.networks[i]} stats:\n{normalizer.get_stats()}\n\n",
                )

        # Machine-loadable stats (one file per network, keyed by network name)
        stats_dict = {}
        for i, normalizer in enumerate(self.data_normalizers):
            stats_dict[self.args.data.networks[i]] = normalizer.get_stats()
        torch.save(stats_dict, os.path.join(log_dir, "normalizer_stats.pt"))


    def configure_optimizers(self):
        if self.args.optimizer.type is None:
            self.args.optimizer.type = "Adam"
        optimizer = getattr(torch.optim, self.args.optimizer.type)
        print(f'{self.args.optimizer.optimizer_params=}')
        if not isinstance(self.args.optimizer.optimizer_params, Mapping):
            self.args.optimizer.optimizer_params = self.args.optimizer.optimizer_params.to_dict()
        self.optimizer = optimizer(
            self.model.parameters(),
            lr=self.args.optimizer.learning_rate,
            **self.args.optimizer.optimizer_params, #unpack all other optim parameters
        )
        scheduler_type = getattr(self.args.optimizer, "scheduler_type", None)
        if scheduler_type is None:
            return {"optimizer": self.optimizer}

        #TODO: add interval handling for scheduler
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)
        if not isinstance(self.args.optimizer.scheduler_params, Mapping):
            self.args.optimizer.scheduler_params = self.args.optimizer.scheduler_params.to_dict()
        self.scheduler = scheduler(
            self.optimizer,
            **self.args.optimizer.scheduler_params
        )
        config_optim = {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "Validation loss",
            },
        }
        return config_optim
