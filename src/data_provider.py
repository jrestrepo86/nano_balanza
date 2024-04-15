import numpy as np
import torch
from torch.utils.data import Dataset

from .nano_model import Resonators


def toColVector(x):
    """
    Change vectors to column vectors
    """
    x = x.reshape(x.shape[0], -1)
    if x.shape[0] < x.shape[1]:
        x = x.T
    x.reshape((-1, 1))
    return x


class data_set(Dataset):
    def __init__(
        self,
        n_samples,
        min_mass=0.0,
        max_mass=30 * 64.1394e-9,
        resonators_n_points=20000,
        resonators_f_ini=4.965e6,
        resonators_f_final=4.975e6,
        resonators_coupling=0.002,
        resonators_Lm=64.1394e-3,
        resonators_Cm=16.0371e-15,
        resonators_Rm=11.42,
        resonators_C0=43.3903e-12,
        normalize_outs=False,
    ):
        super(data_set, self).__init__()

        self.resonator = Resonators(
            n_points=resonators_n_points,
            f_ini=resonators_f_ini,
            f_final=resonators_f_final,
            coupling=resonators_coupling,
            Lm=resonators_Lm,
            Cm=resonators_Cm,
            Rm=resonators_Rm,
            C0=resonators_C0,
        )

        self.normalize = normalize_outs
        self.data_len = int(n_samples)
        self.mass_samples = (
            np.random.rand(n_samples, 3) * (max_mass - min_mass) + min_mass
        )

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        mass = self.mass_samples[index]
        Ysim = self.resonator.simulate(mass[0], mass[1], mass[2])
        if self.normalize:
            Ysim = Ysim / Ysim.max()
        Ysim = torch.tensor(Ysim, dtype=torch.float)
        mass = torch.tensor(mass, dtype=torch.float)
        return mass, Ysim
