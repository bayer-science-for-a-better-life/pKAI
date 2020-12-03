from torch.nn import functional as F
import torch
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl

BATCH_SIZE = 256


class CustomMSE(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(CustomMSE, self).__init__()
        assert reduction in [
            "mean",
            "sum",
        ], 'Please define <reduction> as "mean" or "sum".'
        self.reduction = reduction

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ):  # , weight: torch.Tensor):
        # difference = (weight(y_pred - y_true))**2 #+ (weight2 * y_pred) ** 2
        difference = (y_pred - y_true) ** 2 + y_pred ** 2
        if self.reduction == "mean":
            return torch.mean(difference)
        elif self.reduction == "sum":
            return torch.sum(difference)


class LitDist(LightningModule):
    def __init__(self):
        super().__init__()

        self.batchsize = BATCH_SIZE
        self.lr = 0.00009
        self.weight_decay = 1e-4

        self.trainset = []
        self.valset = []
        self.testset = []

        self.D_in = 4758
        self.D_out = 1

        H = 14000
        dropout = 0.5
        dropout_decay = 0.24

        # self.layers = [nn.Linear(self.D_in, 1)]
        # self.dropouts = [nn.Dropout(p=0.0)]

        self.layers = nn.ModuleList([nn.Linear(self.D_in, H)])
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout)])

        self.n_layers = 2
        nlayers_hidden = (6500, 2100)
        for i in range(self.n_layers):
            out_dim = nlayers_hidden[i]
            dropout = dropout * dropout_decay

            self.layers.append(nn.Linear(H, out_dim))
            self.dropouts.append(nn.Dropout(p=dropout))
            H = out_dim

        self.layers.append(nn.Linear(H, self.D_out))

        # Assigning the layers as class variables (PyTorch requirement).
        for idx, layer in enumerate(self.layers):
            setattr(self, "fc{}".format(idx), layer)

        # Assigning the dropouts as class variables (PyTorch requirement).
        for idx, dropout in enumerate(self.dropouts):
            setattr(self, "drop{}".format(idx), dropout)

        self.criterion = torch.nn.MSELoss()
        self.criterion_train = torch.nn.MSELoss()  # CustomMSE()

        self.avg_loss_train = 0
        self.avg_loss_val = 0

    def forward(self, X):
        for layer, dropout in zip(self.layers, self.dropouts):
            X = F.relu(layer(X))
            X = dropout(X)
        return self.layers[-1](X)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, self.D_in)
        y = y.reshape(-1, 1)

        outputs = self(x)
        loss = self.criterion_train(outputs, y)

        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss)
        logs = {"train_loss": loss}

        return {"loss": loss, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, self.D_in)
        y = y.reshape(-1, 1)

        y_hat = self.forward(x)
        val_loss = self.criterion_train(y_hat, y)

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(-1, self.D_in)
        y = y.reshape(-1, 1)

        y_hat = self.forward(x)
        test_loss = self.criterion(y_hat, y)

        test_diffs = abs((y - y_hat).reshape(-1)).mean()

        return {"test_loss": test_loss, "test_diffs": test_diffs}

    def training_epoch_end(self, outputs):
        avg_loss_train = torch.stack([x["loss"] for x in outputs]).mean()
        self.avg_loss_train = float(torch.sqrt(avg_loss_train))

        to_print = f"{self.current_epoch:<10} {round(self.avg_loss_train, 4)}"
        # print(to_print)

        tensorboard_logs = {"train_eloss": avg_loss_train}
        return {"train_eloss": avg_loss_train, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.avg_loss_val = float(torch.sqrt(avg_loss_val))

        to_print = f"{self.current_epoch:<20} {round(self.avg_loss_val, 4)}"
        print(to_print)

        tensorboard_logs = {"val_eloss": avg_loss_val}
        return {"val_eloss": avg_loss_val, "log": tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss_test = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.avg_loss_test = float(torch.sqrt(avg_loss_test))

        avg_mae_test = torch.stack([abs(x["test_diffs"]) for x in outputs]).mean()
        self.avg_mae_test = float(avg_mae_test)

        return {
            "RMSE": round(self.avg_loss_test, 3),
            "MAE": round(self.avg_mae_test, 3),
        }
