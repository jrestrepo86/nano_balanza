import matplotlib.pyplot as plt
import numpy as np


class Resonators:
    """Simulate 4 resonators single linear coupling model."""

    def __init__(
        self,
        coupling=0.002,
        Lm=64.1394e-3,
        Cm=16.0371e-15,
        Rm=11.42,
        C0=43.3903e-12,
        f_ini=4.965e6,
        f_final=4.975e6,
        n_points=20000,
    ):

        super(Resonators, self).__init__()
        self.Lm = Lm
        self.Cm = Cm
        self.Rm = Rm
        self.C0 = C0
        self.Lk = coupling * Lm
        self.fsim = np.linspace(f_ini, f_final, n_points)

    def simulate(self, mass1, mass2, mass3):
        Zsim = self.model(mass1, mass2, mass3)
        return np.abs(1 / Zsim)

    def model(self, mass1, mass2, mass3):
        w = 2 * np.pi * self.fsim
        b = 1j * w

        Z_Co = 1 / (b * self.C0)  # Impedancia ZCo

        Z_Q1 = (b * self.Lm) + (b * mass1) + self.Rm + (1.0 / (b * self.Cm)) + Z_Co
        Z_Q2 = (b * self.Lm) + (b * mass2) + self.Rm + (1.0 / (b * self.Cm)) + Z_Co
        Z_Q3 = (b * self.Lm) + (b * mass3) + self.Rm + (1.0 / (b * self.Cm)) + Z_Co

        # Calculamos las impedancias series y paralelos
        Z_coupling = -b * self.Lk
        # serie entre Z_Q3 + Zcoupling_Lk9
        Z_half = Z_coupling + Z_Q3
        # paralelo de Z_half con Z_coupling_Lk8
        Z_half = Z_coupling * Z_half / (Z_coupling + Z_half)
        # serie entre paralelo anterior + Zcoupling_Lk7
        Z_half = Z_half + Z_coupling
        # paralelo de Z_Q2 con Z_half
        Z_half = Z_half * Z_Q2 / (Z_half + Z_Q2)
        # serie entre paralelo anterior + Zcoupling_Lk6
        Z_half = Z_half + Z_coupling
        # paralelo de Z_half con Z_coupling_Lk5
        Z_half = Z_half * Z_coupling / (Z_coupling + Z_half)
        # serie entre paralelo anterior + Zcoupling_Lk4
        Z_half = Z_half + Z_coupling
        # paralelo de Z_Q1 con Z_half
        Z_half = Z_half * Z_Q1 / (Z_half + Z_Q1)
        # serie entre paralelo anterior + Zcoupling_Lk3
        Z_half = Z_half + Z_coupling
        # paralelo de Z_half con Z_coupling_Lk2
        Z_half = Z_half * Z_coupling / (Z_coupling + Z_half)
        # serie entre paralelo anterior + Zcoupling_Lk1
        Z_half = Z_half + Z_coupling

        Z_Q0 = (b * self.Lm) + (1.0 / (b * self.Cm)) + self.Rm + Z_half
        # paralelo final
        Z_total = Z_Co * Z_Q0 / (Z_Co + Z_Q0)
        return Z_total


if __name__ == "__main__":
    mass_min, mass_max = 64.1394e-9, 30 * 64.1394e-9
    Mass = np.random.rand(3, 1) * mass_max
    coupling = 0.002
    Res = Resonators(coupling)

    Ysim = Res.simulate(Mass[0], Mass[1], Mass[2])

    plt.plot(Ysim)
    plt.show()
