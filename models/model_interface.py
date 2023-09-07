import torch
import torch.nn.functional as F

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class DTI_prediction(pl.LightningModule):
    def __init__(self, model, len_train_dataloader, learning_rate):
        super().__init__()
        self.model = model
        self.len_train_dataloader = len_train_dataloader
        self.learning_rate = learning_rate
        # self.loss = torch.nn.binary

    def step(self, batch):
        mol_feature, prot_feat_student, prot_feat_teacher, y, source = batch
        prot_feat_teacher = prot_feat_teacher.detach()
        pred, lambda_ = self.model(mol_feature, prot_feat_student, prot_feat_teacher)

        loss = F.binary_cross_entropy_with_logits(pred, y)
        pred = F.sigmoid(pred)
        # loss = F.smooth_l1_loss(pred, y)
        # pred = pred.float()

        return pred, y, source, loss, lambda_

    def training_step(self, batch, batch_idx):
        _, _, _, loss, lambda_ = self.step(batch)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_lambda_", lambda_, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        preds, y, _, loss, lambda_ = self.step(batch)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_lambda_", lambda_, on_step=False, on_epoch=True, prog_bar=True)

        return {"preds": preds, "target": y}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp["preds"] for tmp in outputs], 0).detach().cpu()
        targets = torch.cat([tmp["target"] for tmp in outputs], 0).detach().cpu().long()

        auroc = torchmetrics.functional.auroc(preds, targets.long(), task="binary")
        auprc = torchmetrics.functional.average_precision(
            preds, targets.long(), task="binary"
        )
        self.log("valid_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        preds, y, _, loss, lambda_ = self.step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"preds": preds, "target": y}

    def test_epoch_end(self, outputs):
        preds = torch.cat([tmp["preds"] for tmp in outputs], 0).detach().cpu()
        targets = torch.cat([tmp["target"] for tmp in outputs], 0).detach().cpu().long()

        auroc = torchmetrics.functional.auroc(preds, targets.long(), task="binary")
        auprc = torchmetrics.functional.average_precision(
            preds, targets.long(), task="binary"
        )
        self.log("test_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_auprc", auprc, on_step=False, on_epoch=True, prog_bar=True)

        conf_mat = torchmetrics.functional.confusion_matrix(
            preds, targets, task="binary"
        )

        print(conf_mat)

    def predict_step(self, batch, batch_idx):
        pred, y, source, _, _ = self.step(batch)

        return pred, y, source

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100 * self.len_train_dataloader
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def define_callbacks(PROJECT_NAME):
    callbacks = [
        ModelCheckpoint(
            monitor="valid_auprc",
            mode="max",
            save_top_k=1,
            dirpath=f"weights/{PROJECT_NAME}",
            filename="DTI-{epoch:03d}-{valid_loss:.4f}-{valid_auroc:.4f}-{valid_auprc:.4f}",
        ),
    ]

    return callbacks
