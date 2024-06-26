import numpy as np
import torch
from torch.utils.data import Dataset

from .nano_model import Resonators


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

        self.max_mass = max_mass
        self.min_mass = min_mass
        self.data_len = int(n_samples)
        # self.mass_samples = (
        #     np.random.rand(self.data_len, 3) * (max_mass - min_mass) + min_mass
        # )
        # self.mass_samples = np.random.rand(self.data_len, 3)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        mass = np.random.rand(3)
        m = mass * (self.max_mass - self.min_mass) + self.min_mass
        Ysim = self.resonator.simulate(m[0], m[1], m[2])
        Ysim = torch.tensor(Ysim, dtype=torch.float)
        mass = torch.tensor(mass, dtype=torch.float)
        return mass, Ysim
