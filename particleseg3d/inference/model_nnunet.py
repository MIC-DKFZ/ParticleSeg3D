import pytorch_lightning as pl
from particleseg3d.utils import utils
import torch.nn.functional as F
from torch import nn
from typing import Any
import numpy as np
from os.path import join
from pathlib import Path
import torch
import json
from typing import Tuple, Any, Optional, Dict
from particleseg3d.utils.nnunet_utils import InitWeights_He, Generic_UNet


class Nnunet(pl.LightningModule):
    def __init__(self, model_dir: str, folds: Optional[Tuple[int, int, int, int, int]] = None, nnunet_trainer: str = "nnUNetTrainerV2__nnUNetPlansv2.1", 
             configuration: str = "3D", tta: bool = True, checkpoint: str = "model_best") -> None:
        """
        Initializes the Nnunet class with given parameters and loads the model checkpoint.

        Args:
            model_dir (str): The directory where model checkpoints are stored.
            folds (Tuple[int, int, int, int, int], optional): A tuple containing the folds to be used. Defaults to None, which corresponds to all folds (0, 1, 2, 3, 4).
            nnunet_trainer (str, optional): The name of the nnunet trainer to be used. Defaults to "nnUNetTrainerV2__nnUNetPlansv2.1".
            configuration (str, optional): The type of configuration, either "3D" or "2D". Defaults to "3D".
            tta (bool, optional): Whether to use test-time augmentation. Defaults to True.
            checkpoint (str, optional): The name of the checkpoint to be loaded. Can be either "model_best" or "model_final_checkpoint". Defaults to "model_best".
        """
        super().__init__()

        self.nnunet_trainer = nnunet_trainer
        self.configuration = configuration
        self.network = self.load_checkpoint(model_dir, folds, configuration, checkpoint)
        self.final_activation = nn.Softmax(dim=2)
        self.tta = tta

    def load_checkpoint(self, model_dir: str, folds: Optional[Tuple[int, int, int, int, int]], configuration: str, checkpoint: str) -> nn.ModuleList:
        """
        Loads the model checkpoints for the given folds.

        Args:
            model_dir (str): The directory where model checkpoints are stored.
            folds (Tuple[int, int, int, int, int], optional): A tuple containing the folds to be used. If None, it corresponds to all folds (0, 1, 2, 3, 4).
            configuration (str): The type of configuration, either "3D" or "2D".
            checkpoint (str): The name of the checkpoint to be loaded. Can be either "model_best" or "model_final_checkpoint".

        Returns:
            nn.ModuleList: A list of loaded model checkpoints, one for each fold.

        Raises:
            RuntimeError: If no folds are found in the model directory.
        """
        ensemble = []
        if folds is None:
            folds = (0, 1, 2, 3, 4)
        folds = ["fold_{}".format(fold) for fold in folds]
        for fold in folds:
            checkpoint_path = join(model_dir, fold, "{}.model".format(checkpoint))
            if Path(checkpoint_path).is_file():
                with open(join(model_dir, fold, "debug.json")) as f:
                    model_config = json.load(f)
                network = self.initialize_network(model_config, configuration)
                network.load_state_dict(torch.load(checkpoint_path)["state_dict"])
                ensemble.append(network)
            else:
                print("Could not find fold {} for ensemble.".format(fold))
        if not ensemble:
            raise RuntimeError("Could not find any folds in experiment_dir ({}).".format(model_dir))
        ensemble = nn.ModuleList(ensemble)
        return ensemble

    def initialize_network(self, model_config: Dict[str, Any], configuration: str) -> Generic_UNet:
        """
        Initializes the network using the provided model configuration.

        Args:
            model_config (Dict[str, Any]): The configuration of the model, usually loaded from a JSON file.
            configuration (str): The type of configuration, either "3D" or "2D".

        Returns:
            Generic_UNet: The initialized network.

        Raises:
            RuntimeError: If the configuration type is not supported.
        """
        if configuration == "3d_fullres":
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            if self.nnunet_trainer == "nnUNetTrainerV2_BN__nnUNetPlansv2.1":
                norm_op = nn.BatchNorm3d
            else:
                norm_op = nn.InstanceNorm3d
        elif configuration == "2d":
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d
        else:
            raise RuntimeError("Configuration not supported.")

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        net_num_pool_op_kernel_sizes = json.loads(model_config["net_num_pool_op_kernel_sizes"])
        net_conv_kernel_sizes = json.loads(model_config["net_conv_kernel_sizes"])
        network = Generic_UNet(int(model_config["num_input_channels"]), int(model_config["base_num_features"]), int(model_config["num_classes"]),
                                    len(net_num_pool_op_kernel_sizes),
                                    int(model_config["conv_per_stage"]), 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
        network.inference_apply_nonlin = lambda x: F.softmax(x, 1)
        return network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        y = [network(x) for network in self.network]  # e, (b, c, x, y, z) -> (e, b, c, x, y, z)
        y = torch.stack(y)
        y = torch.permute(y, (1, 0, 2, 3, 4, 5))  # (e, b, c, x, y, z) -> (b, e, c, x, y, z)
        return y

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the optimizer and the learning rate scheduler.

        Returns:
            Dict[str, Any]: A dictionary containing the optimizer, the learning rate scheduler, and the monitor metric.
        """
        optimizer = utils.create_optimizer(self.config['optimizer'], self)
        lr_scheduler = utils.create_lr_scheduler(self.config.get('lr_scheduler', None), optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "Val/Mean Class Dice"}

    def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Performs a single step in the training loop.

        Args:
            train_batch (Tuple[torch.Tensor, torch.Tensor]): The current training batch, consisting of input data and target labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss for the current training step.
        """
        x, y = train_batch
        output = self(x)
        loss = self.loss_criterion(output, y)
        self.log('Train/Loss', loss)
        return loss

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Performs a single step in the validation loop.

        Args:
            val_batch (Tuple[torch.Tensor, torch.Tensor]): The current validation batch, consisting of input data and target labels.
            batch_idx (int): The index of the current batch.
        """
        x, y = val_batch
        output = self(x)
        mean_dice, class_dices = self.eval_criterion(output, y)
        self.log('Val/Mean Class Dice', mean_dice)
        for i, class_dice in enumerate(class_dices):
            self.log('Val/Class Dice {}'.format(i), class_dice)

    def prediction_setup(self, aggregator: Any, chunked: bool, zscore: Dict[str, float]) -> None:
        """
        Sets up the model for prediction.

        Args:
            aggregator (Any): The aggregator to be used for predictions.
            chunked (bool): A flag indicating whether chunked prediction should be used.
            zscore (Dict[str, float]): A dictionary containing the mean and standard deviation for z-score normalization.

        Returns:
            None
        """
        self.aggregator = aggregator
        self.chunked = chunked
        self.zscore = zscore

    def predict_step(self, batch: Any, batch_idx: int) -> bool:
        """
        Performs a single step in the prediction loop.

        Args:
            batch (Any): The current batch, consisting of an image patch and its indices.
            batch_idx (int): The index of the current batch.

        Returns:
            bool: Always True. Indicates that the prediction step was successful.
        """
        img_patch, patch_indices = batch
        img_patch -= self.zscore["mean"]
        img_patch /= self.zscore["std"]
        if not self.tta:
            pred_patch = self(img_patch)
            pred_patch = self.final_activation(pred_patch)
        else:
            pred_patch = self.predict_with_tta(img_patch)
        pred_patch = torch.mean(pred_patch, axis=1)  # (b, e, c, x, y, z) -> (b, c, x, y, z)
        pred_patch = pred_patch.cpu().numpy()
        patch_indices = [value.cpu().numpy() for value in patch_indices]
        for i in range(len(pred_patch)):
            if self.chunked:
                self.aggregator.append(pred_patch[i], (patch_indices[0][i], patch_indices[1][i]))
            else:
                self.aggregator.append(pred_patch[i], patch_indices[i])
        return True

    def predict_with_tta(self, img_patch: torch.Tensor) -> torch.Tensor:
        """
        Performs prediction with Test Time Augmentation (TTA).

        Args:
            img_patch (torch.Tensor): The image patch to predict.

        Returns:
            torch.Tensor: The prediction for the image patch.
        """
        flips = [(4, ), (3, ), (4, 3), (2, ), (4, 2), (3, 2), (4, 3, 2)]  # (b, e, c, x, y, z)

        pred_patch = self(img_patch)
        pred_patch = self.final_activation(pred_patch)
        pred_patch = pred_patch / (len(flips) + 1)

        for flip in flips:
            img_patch_flipped = torch.flip(img_patch, flip)
            pred_patch_flipped = self(img_patch_flipped)
            pred_patch_flipped = self.final_activation(pred_patch_flipped)
            pred_patch += torch.flip(pred_patch_flipped, tuple(np.array(flip)+1)) / (len(flips) + 1)

        return pred_patch
