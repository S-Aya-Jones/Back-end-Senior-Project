
class Radiation:
    def __init__(self):
        self.sigma = 5.67 * 10 ^ 8
        self.T_sun = 5800  # k
        self.T_home = 300  # k
        self.A_sun = 6.07 * 10 ^ 18  # m2
        self.e_sun = 1

    def DQDT_Radiation(self):
        return self.e_sun * self.sigma * self.A_sun * (pow(self.T_sun, 4) - pow(self.T_home, 4))







