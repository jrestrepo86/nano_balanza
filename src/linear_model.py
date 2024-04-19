import numpy as np
import schedulefree
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .layers import Regressor


class LinearModel(nn.Module):
    """docstring for Model."""

    def __init__(
        self,
        input_dim,
        reg_inner_layers=[4096, 1024, 512, 128, 64, 16],
        reg_afn="gelu",
        reg_dropout=0.1,
        device=None,
    ):
        super(LinearModel, self).__init__()
        # select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        torch.device(self.device)

        self.regressor = Regressor(
            input_dim,
            inner_layers=reg_inner_layers,
            afn=reg_afn,
            dropout=reg_dropout,
        ).to(self.device)

    def forward(self, espectro):
        mass = self.regressor(espectro)
        return mass

    def model_state(self, state="train"):
        if state == "train":
            self.regressor.train()
            self.opt_regressor.train()
        if state == "eval":
            self.regressor.eval()
            self.opt_regressor.eval()

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"model saved: {path}")

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print(f"model loaded: {path}")

    def fit(
        self,
        train_dataset,
        val_dataset,
        batch_size=64,
        max_epochs=2000,
        lr=1e-4,
        verbose=False,
    ):

        # schedule free optimizers
        self.opt_regressor = schedulefree.AdamWScheduleFree(
            self.regressor.parameters(), lr=lr
        )
        # self.opt_regressor = schedulefree.SGDScheduleFree(
        #     self.regressor.parameters(), lr=lr
        # )

        # loss
        regressor_loss_fn = torch.nn.MSELoss()

        # Data
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # training
        train_loss_epoch, val_loss_epoch = [], []
        for i in tqdm(range(max_epochs), disable=not verbose):
            train_loss = []
            self.model_state("train")
            with torch.set_grad_enabled(True):
                for _, (mass, x) in enumerate(self.train_loader):
                    self.opt_regressor.zero_grad()
                    x, mass = x.to(self.device), mass.to(self.device)
                    est_mass = self.forward(x)
                    loss = regressor_loss_fn(mass, est_mass)
                    train_loss.append(loss)
                    loss.backward()
                    self.opt_regressor.step()

                train_loss_epoch.append(torch.stack(train_loss).mean().item())

            # eval
            self.model_state("eval")
            val_loss = []
            with torch.no_grad():
                for _, (mass, x) in enumerate(self.val_loader):
                    x, mass = x.to(self.device), mass.to(self.device)
                    est_mass = self.forward(x)
                    loss = regressor_loss_fn(mass, est_mass)
                    val_loss.append(loss)

                val_loss = torch.stack(val_loss).mean().item()
                val_loss_epoch.append(val_loss)
            text = f"Epoca {i+1}/{max_epochs} |" + f" reg_loss={val_loss} |"
            print(text)

        self.val_loss_epoch = np.array(val_loss_epoch)
        self.train_loss_epoch = np.array(train_loss_epoch)

    def get_curves(self):
        return self.val_loss_epoch, self.train_loss_epoch

    def test_model(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=1)
        reg_loss_fn = torch.nn.MSELoss()

        self.model_state("eval")
        mass_array, est_mass_array, test_loss = [], [], []
        with torch.no_grad():
            for _, (mass, x) in enumerate(test_loader):
                x, mass = x.to(self.device), mass.to(self.device)
                est_mass = self.forward(x)
                loss = reg_loss_fn(mass, est_mass)
                test_loss.append(loss.item())
                mass_array.append(mass.detach().cpu())
                est_mass_array.append(est_mass.detach().cpu())

        test_loss = np.array(test_loss)
        test_mass = np.array(mass_array)
        test_est_mass = np.array(est_mass_array)
        return test_loss, test_mass, test_est_mass
