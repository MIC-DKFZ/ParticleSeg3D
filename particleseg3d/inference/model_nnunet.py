import pytorch_lightning as pl
from particleseg3d.utils import utils
from nnunet.network_architecture.generic_UNet import Generic_UNet
import torch.nn.functional as F
from torch import nn
from nnunet.network_architecture.initialization import InitWeights_He
from typing import Any
import numpy as np
from os.path import join
from pathlib import Path
import torch
import json


class Nnunet(pl.LightningModule):
    def __init__(self, model_dir, folds=None, nnunet_trainer="nnUNetTrainerV2__nnUNetPlansv2.1", configuration="3D", tta=True, checkpoint="model_best"):  # checkpoint: model_best, model_final_checkpoint
        super().__init__()

        self.nnunet_trainer = nnunet_trainer
        self.configuration = configuration
        self.network = self.load_checkpoint(model_dir, folds, configuration, checkpoint)
        self.final_activation = nn.Softmax(dim=2)
        self.tta = tta

    def load_checkpoint(self, model_dir, folds, configuration, checkpoint):
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

    def initialize_network(self, model_config, configuration):
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

    def forward(self, x):
        y = [network(x) for network in self.network]  # e, (b, c, x, y, z) -> (e, b, c, x, y, z)
        y = torch.stack(y)
        y = torch.permute(y, (1, 0, 2, 3, 4, 5))  # (e, b, c, x, y, z) -> (b, e, c, x, y, z)
        return y

    def configure_optimizers(self):
        optimizer = utils.create_optimizer(self.config['optimizer'], self)
        lr_scheduler = utils.create_lr_scheduler(self.config.get('lr_scheduler', None), optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "Val/Mean Class Dice"}

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self(x)
        loss = self.loss_criterion(output, y)
        self.log('Train/Loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        output = self(x)
        mean_dice, class_dices = self.eval_criterion(output, y)
        self.log('Val/Mean Class Dice', mean_dice)
        for i, class_dice in enumerate(class_dices):
            self.log('Val/Class Dice {}'.format(i), class_dice)

    def prediction_setup(self, aggregator, chunked, zscore):
        self.aggregator = aggregator
        self.chunked = chunked
        self.zscore = zscore

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
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

    def predict_with_tta(self, img_patch):
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
