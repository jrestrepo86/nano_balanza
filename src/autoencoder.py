# scheduler
# pip install schedulefree
import numpy as np
import schedulefree
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_activation_fn(afn):
    activation_functions = {
        "linear": lambda: lambda x: x,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
        "leaky_relu": nn.LeakyReLU,
        "threshold": nn.Threshold,
        "hardtanh": nn.Hardtanh,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "log_sigmoid": nn.LogSigmoid,
        "softplus": nn.Softplus,
        "softshrink": nn.Softshrink,
        "softsign": nn.Softsign,
        "tanhshrink": nn.Tanhshrink,
        "softmax": nn.Softmax,
        "gelu": nn.GELU,
    }

    if afn not in activation_functions:
        raise ValueError(
            f"'{afn}' is not included in activation_functions. Use below one \n {activation_functions.keys()}"
        )

    return activation_functions[afn]


class Regressor(nn.Module):
    def __init__(self, feature_dim, inner_layers, afn="gelu", dropout=0.1):
        super(Regressor, self).__init__()
        activation_fn = get_activation_fn(afn)
        inner_layers = [int(hl) for hl in inner_layers]
        # Model
        seq = [nn.Linear(feature_dim, inner_layers[0]), activation_fn()]
        for i in range(len(inner_layers) - 1):
            seq += [
                nn.Linear(inner_layers[i], inner_layers[i + 1]),
                activation_fn(),
            ]
        seq += [nn.Linear(inner_layers[-1], 3), get_activation_fn("relu")()]
        self.model = nn.Sequential(*seq)
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        return self.model(features)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        features_dim,
        inner_layers,
        afn="gelu",
        dropout=0.1,
    ):
        super(AutoEncoder, self).__init__()

        activation_fn = get_activation_fn(afn)
        encoder_inner_layers = [int(hl) for hl in inner_layers]
        decoder_inner_layers = [int(hl) for hl in inner_layers[::-1]]
        # Encoder
        encoder_seq = [nn.Linear(input_dim, encoder_inner_layers[0]), activation_fn()]
        for i in range(len(encoder_inner_layers) - 1):
            encoder_seq += [
                nn.Linear(encoder_inner_layers[i], encoder_inner_layers[i + 1]),
                activation_fn(),
            ]
        encoder_seq += [nn.Linear(encoder_inner_layers[-1], features_dim)]
        self.Encoder = nn.Sequential(*encoder_seq)
        # Dencoder
        decoder_seq = [
            nn.Linear(features_dim, decoder_inner_layers[0]),
            activation_fn(),
        ]
        for i in range(len(decoder_inner_layers) - 1):
            decoder_seq += [
                nn.Linear(decoder_inner_layers[i], decoder_inner_layers[i + 1]),
                activation_fn(),
            ]
        decoder_seq += [
            nn.Linear(decoder_inner_layers[-1], input_dim),
            get_activation_fn("relu")(),
        ]
        self.Decoder = nn.Sequential(*decoder_seq)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # positional embedding
        self.PositionalEmbedding(input_dim)

    def PositionalEmbedding(self, input_dim, n=10000.0):
        assert (
            input_dim % 2 == 0
        ), f"Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={input_dim})"
        pe = torch.zeros(input_dim, dtype=torch.float, requires_grad=False)
        positions = torch.arange(0, input_dim, dtype=torch.float)

        # 10000^(2i/model_dim), i is the index of embedding
        denominators = torch.pow(n, 2 * torch.arange(0, input_dim // 2) / input_dim)
        # sin(pos/10000^(2i/model_dim))
        pe[0::2] = torch.sin(positions[0::2] / denominators)
        # cos(pos/10000^(2i/model_dim))
        pe[1::2] = torch.cos(positions[0::2] / denominators)
        # pe shape (1, seq_len, model_dim)
        self.position_embedding = pe.unsqueeze(0)

    def forward(self, espectro, just_encoder=False):
        pe = espectro + self.position_embedding.to(espectro.device)
        features = self.Encoder(pe)
        if just_encoder:
            out_dec = []
        else:
            out_dec = self.Decoder(features)
        return out_dec, features


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

        self.reg = Regressor(
            features_dim,
            inner_layers=reg_inner_layers,
            afn=reg_afn,
            dropout=reg_dropout,
        ).to(self.device)

    def forward(self, espectro, just_encoder=False):
        dec_out, features = self.autoEncoder(espectro, just_encoder)
        mass = self.reg(features)
        return dec_out, mass

    def model_state(self, state="train"):
        if state == "train":
            self.autoEncoder.train()
            self.reg.train()
        if state == "eval":
            self.autoEncoder.eval()
            self.reg.eval()

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print("model saved")

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print("model loaded")
        self.eval()

    def fit(
        self,
        train_dataset,
        val_dataset,
        batch_size=64,
        max_epochs=2000,
        lr=1e-4,
        verbose=False,
        encoder_weight=0.5,
        reg_weight=0.5,
    ):

        # schedule free optimizers
        opt_enc = schedulefree.AdamWScheduleFree(self.autoEncoder.parameters(), lr=lr)
        opt_reg = schedulefree.AdamWScheduleFree(self.reg.parameters(), lr=lr)

        # loss
        enc_loss_fn = torch.nn.MSELoss()
        reg_loss_fn = torch.nn.MSELoss()

        # Data
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # training
        train_loss_epoch, train_reg_loss_epoch, train_enc_loss_epoch = [], [], []
        val_loss_epoch, val_reg_loss_epoch, val_enc_loss_epoch = [], [], []
        for i in tqdm(range(max_epochs), disable=not verbose):
            self.model_state("train")
            with torch.set_grad_enabled(True):
                train_loss, train_reg_loss, train_enc_loss = [], [], []
                for _, (mass, x) in enumerate(self.train_loader):
                    opt_enc.zero_grad()
                    opt_reg.zero_grad()
                    x = x.to(self.device)
                    mass = mass.to(self.device)
                    dec_out, est_mass = self.forward(x)
                    loss_enc = enc_loss_fn(dec_out, x)
                    loss_reg = reg_loss_fn(mass, est_mass)
                    loss = reg_weight * loss_reg + encoder_weight * loss_enc
                    train_loss.append(loss)
                    train_reg_loss.append(loss_reg)
                    train_enc_loss.append(loss_enc)
                    loss.backward()
                    opt_enc.step()
                    opt_reg.step()

                train_loss_epoch.append(torch.stack(train_loss).mean().item())
                train_reg_loss_epoch.append(torch.stack(train_reg_loss).mean().item())
                train_enc_loss_epoch.append(torch.stack(train_enc_loss).mean().item())

            # eval
            self.model_state("eval")
            with torch.no_grad():
                val_loss, val_reg_loss, val_enc_loss = [], [], []
                for _, (mass, x) in enumerate(self.val_loader):
                    x = x.to(self.device)
                    mass = mass.to(self.device)
                    dec_out, est_mass = self.forward(x)
                    loss_enc = enc_loss_fn(dec_out, x)
                    loss_reg = reg_loss_fn(mass, est_mass)
                    loss = reg_weight * loss_reg + encoder_weight * loss_enc
                    val_loss.append(loss)
                    val_reg_loss.append(loss_reg)
                    val_enc_loss.append(loss_enc)

                val_loss = torch.stack(val_loss).mean().item()
                val_reg_loss = torch.stack(val_reg_loss).mean().item()
                val_enc_loss = torch.stack(val_enc_loss).mean().item()
                val_loss_epoch.append(val_loss)
                val_reg_loss_epoch.append(val_reg_loss)
                val_enc_loss_epoch.append(val_enc_loss)
            text = (
                f"Epoca {i}/{max_epochs} |"
                + f"reg_loss={val_reg_loss} |"
                + f"enc_loss={val_enc_loss} |"
                + f"loss={val_loss}"
            )
            print(text)

        self.val_loss_epoch = np.array(val_loss_epoch)
        self.val_reg_loss_epoch = np.array(val_reg_loss_epoch)
        self.val_enc_loss_epoch = np.array(val_enc_loss_epoch)
        self.train_loss_epoch = np.array(train_loss_epoch)
        self.train_reg_loss_epoch = np.array(train_reg_loss_epoch)
        self.train_enc_loss_epoch = np.array(train_enc_loss_epoch)

    def get_curves(self, all=False):
        if all:
            return (
                self.train_loss_epoch,
                self.train_reg_loss_epoch,
                self.train_enc_loss_epoch,
                self.val_loss_epoch,
                self.val_reg_loss_epoch,
                self.val_enc_loss_epoch,
            )
        else:
            return self.val_loss_epoch, self.val_reg_loss_epoch, self.val_enc_loss_epoch

    def test_model(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=1)
        reg_loss_fn = torch.nn.MSELoss()

        self.model_state("eval")
        mass_array, est_mass_array, test_reg_loss = [], [], []

        with torch.no_grad():
            for _, (mass, x) in enumerate(test_loader):
                x = x.to(self.device)
                mass = mass.to(self.device)
                _, est_mass = self.forward(x, just_encoder=True)
                loss_reg = reg_loss_fn(mass, est_mass)
                test_reg_loss.append(loss_reg.item())
                mass_array.append(mass.detach().cpu())
                est_mass_array.append(est_mass.detach().cpu())

        test_reg_loss = np.array(test_reg_loss)
        test_mass = np.array(mass_array)
        test_est_mass = np.array(est_mass_array)
        return test_reg_loss, test_mass, test_est_mass
