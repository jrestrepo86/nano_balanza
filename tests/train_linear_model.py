import os
import sys
import numpy as np

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.linear_model import LinearModel
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
}
nsamples = 2048 * 10
training_data = resonator_data(nsamples, **data_parameters)
val_data = resonator_data(0.2 * nsamples, **data_parameters)

# autoencoder model
nfeatures = 24
model_parameters = {
    "input_dim": data_parameters["resonators_n_points"],
    "reg_inner_layers": [4096, 1024, 512, 128, 64, 16],
    "reg_afn": "gelu",
    "reg_dropout": 0.1,
    "device": "cuda",
    # "device": "cpu",
}
model = LinearModel(**model_parameters)

# Entrenar modelo
training_parameters = {
    "batch_size": 512,
    "max_epochs": 10,
    "lr": 1e-4,
    "verbose": False,
}
print("Entrenando sin ruido")
model.fit(training_data, val_data, **training_parameters)
model_save_file = "../models/liner_model_01"
model.save_model(model_save_file)

val_loss, train_loss = model.get_curves()


print(f"train_loss={train_loss[-1]} | val_loss={val_loss[-1]}")

fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)
axs.semilogy(train_loss, label="train")
axs.semilogy(val_loss, label="val")
axs.legend(loc="upper right")
fig.suptitle("Linear Model Validation losses")

# testing
test_data = resonator_data(0.2 * nsamples, **data_parameters)  # crear datos test
test_loss, mass, est_mass = model.test_model(test_data)  # hacer test
print(
    f"test_reg_loss_median={np.median(test_loss)}, test_reg_loss_std={test_loss.std()}"
)

fig, axs = plt.subplots(1, 2, sharex=False, sharey=False)
axs[0].hist(test_loss, bins=32)
axs[1].boxplot(test_loss)
fig.suptitle("Test Reg loss with Noise")
plt.show()
