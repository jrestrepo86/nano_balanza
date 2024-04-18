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


def PositionalEmbedding(input_dim, n=10000.0):
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
    return pe.unsqueeze(0)


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
        self.positional_embedding = PositionalEmbedding(input_dim)

    def forward(self, espectro):
        features = self.encoder_forward(espectro)
        out_dec = self.Decoder(features)
        return out_dec, features

    def encoder_forward(self, espectro):
        espectro = espectro + self.positional_embedding.to(espectro.device)
        return self.Encoder(espectro)
