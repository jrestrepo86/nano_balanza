import numpy as np
import schedulefree
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .layers import AutoEncoder, Regressor


class Model(nn.Module):
    """docstring for Model."""

    def __init__(
        self,
        input_dim,
        features_dim,
        encoder_inner_layers=[128, 64, 32],
        encoder_afn="gelu",
        encoder_dropout=0.1,
        reg_inner_layers=[128, 64, 32],
        reg_afn="gelu",
        reg_dropout=0.1,
        device=None,
    ):
        super(Model, self).__init__()
        # select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        torch.device(self.device)

        self.autoEncoder = AutoEncoder(
            input_dim,
            features_dim,
            inner_layers=encoder_inner_layers,
            afn=encoder_afn,
            dropout=encoder_dropout,
        ).to(self.device)

        self.regressor = Regressor(
            features_dim,
            inner_layers=reg_inner_layers,
            afn=reg_afn,
            dropout=reg_dropout,
        ).to(self.device)

    def forward(self, espectro):
        dec_out, features = self.autoEncoder(espectro)
        mass = self.regressor(features)
        return dec_out, mass

    def model_state(self, state="train"):
        if state == "train":
            self.regressor.train()
            self.opt_regressor.train()
            self.autoEncoder.train()
            self.opt_autoencoder.train()
        if state == "eval":
            self.autoEncoder.eval()
            self.regressor.eval()
            self.opt_autoencoder.eval()
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
        self.opt_autoencoder = schedulefree.AdamWScheduleFree(
            self.autoEncoder.parameters(), lr=10.0 * lr
        )
        self.opt_regressor = schedulefree.AdamWScheduleFree(
            self.regressor.parameters(), lr=lr
        )
        # self.opt_encoder = schedulefree.AdamWScheduleFree(
        #     self.autoEncoder.Encoder.parameters(), lr=lr
        # )

        # self.opt_enc = schedulefree.SGDScheduleFree(
        #     self.autoEncoder.parameters(), lr=lr
        # )
        # self.opt_regressor = schedulefree.SGDScheduleFree(
        #     self.regressor.parameters(), lr=lr
        # )

        # loss
        autoencoder_loss_fn = torch.nn.MSELoss()
        regressor_loss_fn = torch.nn.L1Loss()
        # regressor_loss_fn = torch.nn.MSELoss()

        # Data
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # training
        train_reg_loss_epoch, train_enc_loss_epoch = [], []
        val_reg_loss_epoch, val_enc_loss_epoch = [], []
        for i in tqdm(range(max_epochs), disable=not verbose):
            train_reg_loss, train_enc_loss = [], []
            self.model_state("train")
            with torch.set_grad_enabled(True):
                for _, (mass, x) in enumerate(self.train_loader):
                    x, mass = x.to(self.device), mass.to(self.device)
                    dec_out, est_mass = self.forward(x)
                    loss_enc = autoencoder_loss_fn(dec_out, x)
                    loss_reg = regressor_loss_fn(mass, est_mass)
                    loss = loss_enc + loss_reg
                    train_enc_loss.append(loss_enc)
                    train_reg_loss.append(loss_reg)
                    self.opt_autoencoder.zero_grad()
                    self.opt_regressor.zero_grad()
                    loss.backward()
                    self.opt_autoencoder.step()
                    self.opt_autoencoder.step()

                train_enc_loss_epoch.append(torch.stack(train_enc_loss).mean().item())
                train_reg_loss_epoch.append(torch.stack(train_reg_loss).mean().item())

            # eval
            self.model_state("eval")
            val_reg_loss, val_enc_loss = [], []
            with torch.no_grad():
                for _, (mass, x) in enumerate(self.val_loader):
                    x, mass = x.to(self.device), mass.to(self.device)
                    dec_out, est_mass = self.forward(x)
                    loss_enc = autoencoder_loss_fn(dec_out, x)
                    loss_reg = regressor_loss_fn(mass, est_mass)
                    val_reg_loss.append(loss_reg)
                    val_enc_loss.append(loss_enc)

                val_reg_loss = torch.stack(val_reg_loss).mean().item()
                val_enc_loss = torch.stack(val_enc_loss).mean().item()
                val_reg_loss_epoch.append(val_reg_loss)
                val_enc_loss_epoch.append(val_enc_loss)
            text = (
                f"Epoca {i+1}/{max_epochs} |"
                + f" reg_loss={val_reg_loss} |"
                + f" enc_loss={val_enc_loss}"
            )
            print(text)

        self.val_reg_loss_epoch = np.array(val_reg_loss_epoch)
        self.val_enc_loss_epoch = np.array(val_enc_loss_epoch)
        self.train_reg_loss_epoch = np.array(train_reg_loss_epoch)
        self.train_enc_loss_epoch = np.array(train_enc_loss_epoch)

    def get_curves(self, all=False):
        if all:
            return (
                self.train_reg_loss_epoch,
                self.train_enc_loss_epoch,
                self.val_reg_loss_epoch,
                self.val_enc_loss_epoch,
            )
        else:
            return self.val_reg_loss_epoch, self.val_enc_loss_epoch

    def test_model(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=1)
        reg_loss_fn = torch.nn.MSELoss()

        self.model_state("eval")
        mass_array, est_mass_array, test_reg_loss = [], [], []

        with torch.no_grad():
            for _, (mass, x) in enumerate(test_loader):
                x = x.to(self.device)
                mass = mass.to(self.device)
                features = self.autoEncoder.encoder_forward(x)
                est_mass = self.regressor(features)
                loss_reg = reg_loss_fn(mass, est_mass)
                test_reg_loss.append(loss_reg.item())
                mass_array.append(mass.detach().cpu())
                est_mass_array.append(est_mass.detach().cpu())

        test_reg_loss = np.array(test_reg_loss)
        test_mass = np.array(mass_array)
        test_est_mass = np.array(est_mass_array)
        return test_reg_loss, test_mass, test_est_mass
