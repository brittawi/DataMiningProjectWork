from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    RandAffined,
    Rand3DElasticd,
    DivisiblePadd,
    RandRotated,
    RandGaussianNoised,
    NormalizeIntensityd
)
from monai.networks.layers import Norm

from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
)

import torch
from torch.utils.data import ConcatDataset
import lightning

import numpy as np

class Net(lightning.LightningModule):
    def __init__(self, device, data, augmentation = True, in_channels = 1, out_channels = 2):
        super().__init__()

        self._model = UNet(
            spatial_dims=3,
            in_channels=in_channels, # hard labeling
            out_channels=out_channels, # soft labeling
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout =0.2,
            norm=Norm.BATCH,
            ).to(device)

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=4)
        self.post_label = AsDiscrete(to_onehot=4)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs = 1300
        self.check_val = 30
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_train_loss = []
        self.epoch_val_loss = []
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.augmentation = augmentation
        self.data = data
    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # prepare data
        #data_dir = "C:\\awilde\\britta\\LTU\\DataMining\\Data\\Task02_Heart\\"
        #split_json = "dataset.json"
        #datasets = self.data_dir + split_json
        # load training data
        #train_files = load_decathlon_datalist(self.data, True, "training")
        """
        # try to load validation and test data, if not possible split the training data
        try:
            val_files = load_decathlon_datalist(self.data, True, "validation")
            test_files = load_decathlon_datalist(self.data, True, "test")
        
        except:
            # TODO split is hardcoded, add params
            train_size = int(len(train_files) * 0.8)
            val_size = int(len(train_files) * 0.10)
            #test_size = len(train_files) - train_size - val_size

            val_files = train_files[train_size:train_size+val_size]
            test_files = train_files[train_size+val_size:]
            train_files = train_files[:train_size]
        """
        train_transforms = Compose(
            [   LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                CropForegroundd(keys=["image", "label"], source_key="image"),        
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                DivisiblePadd(["image", "label"], 16),
                NormalizeIntensityd(keys=["image"])
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                DivisiblePadd(["image", "label"], 16),
                NormalizeIntensityd(keys=["image"])
            ]
        )
        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                DivisiblePadd(["image", "label"], 16),
                NormalizeIntensityd(keys=["image"])
            ]
        )

        # Data augmentation
        augm_transforms = Compose(
            [
            RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10), 
            RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),
            RandGaussianNoised(keys='image', prob=0.5),
            NormalizeIntensityd(keys=["image"]),
            ]
        )

        train_ds = CacheDataset(
            data=self.data["training"],
            transform=train_transforms,
            num_workers=12,
        )
        self.val_ds = CacheDataset(
            data=self.data["validation"],
            transform=val_transforms,
            num_workers=12,
        )

        if self.augmentation:
            augm_ds = CacheDataset(
                data = self.data["training"],
                transform=[train_transforms, augm_transforms],
            )

            self.train_ds = ConcatDataset([train_ds, augm_ds])
        else:
            self.train_ds = train_ds

        self.test_ds = CacheDataset(
            data=self.data["test"],
            transform=test_transforms
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            collate_fn=list_data_collate,
            persistent_workers=True
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        return val_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(self.test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        return test_loader

    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4, weight_decay=1e-5)
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4, amsgrad = True)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = (batch["image"].to(self.device), batch["label"].to(self.device))
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        self.training_step_outputs.append(loss)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x for x in self.training_step_outputs]).mean()
        self.epoch_train_loss.append(avg_loss.detach().cpu().numpy())
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        images, labels =  (batch["image"].to(self.device), batch["label"].to(self.device))
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current val_loss: {mean_val_loss} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.metric_values.append(mean_val_dice)
        self.validation_step_outputs.clear()  # free memory
        self.epoch_val_loss.append(mean_val_loss.detach().cpu().numpy())
        # log avg loss for early stopping
        self.log("val_dice", mean_val_dice)
        self.log("val_loss", mean_val_loss)
        return {"log": tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        images, labels =  (batch["image"].to(self.device), batch["label"].to(self.device))
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        self.test_step_outputs.append(loss)
    
    def on_test_epoch_end(self):
        avg_loss = torch.stack([x for x in self.test_step_outputs]).mean()
        avg_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        print(f"Loss on test set: {avg_loss}, dice metric on test set: {avg_dice}")
