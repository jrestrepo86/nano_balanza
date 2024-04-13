import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import autoencoder
from src.data_provider import data_set as resonator_data

# data sets
data_parameters = {
    "min_mass": 0.0,
    "max_mass": 30 * 64.1394e-9,
    "resonators_n_points": 20000,
    "resonators_f_ini": 4.965e6,
    "resonators_f_final": 4.975e6,
    "resonators_coupling": 0.002,
    "resonators_Lm": 64.1394e-3,
    "resonators_Cm": 16.0371e-15,
    "resonators_Rm": 11.42,
    "resonators_C0": 43.3903e-12,
    "normalize_outs": False,
}
nsamples = 5000
training_data = resonator_data(nsamples, **data_parameters)
val_data = resonator_data(nsamples, **data_parameters)
test_data = resonator_data(nsamples, **data_parameters)

# autoencoder model
nfeatures = 24
model_parameters = {
    "input_dim": data_parameters["resonators_n_points"],
    "features_dim": nfeatures,
    "encoder_inner_layers": [2048, 512, 128, 64],
    "encoder_afn": "gelu",
    "encoder_dropout": 0.1,
    "reg_inner_layers": [16, 8],
    "reg_afn": "gelu",
    "reg_dropout": 0.1,
    "device": "cpu",
}
autoencoder = autoencoder.Model(**model_parameters)

# training
training_parameters = {
    "batch_size": 64,
    "max_epochs": 20,
    "lr": 1e-5,
    "weight_decay": 5e-5,
    "stop_patience": 100,
    "stop_min_delta": 0,
    "encoder_weight": 0.5,
    "reg_weight": 0.5,
    "verbose": True,
}
autoencoder.fit(training_data, val_data, **training_parameters)
loss, reg_loss, enc_loss = autoencoder.get_curves()


print(f"reg_loss={reg_loss[-1]}")
# plt.plot(loss)
plt.plot(reg_loss)
# plt.plot(enc_loss)
plt.show()
