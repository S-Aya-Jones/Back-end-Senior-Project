class Conduction:
    def __init__(self):
        self.T_inside = 294  # k
        self.T_outside = 277  # k
        self.Brick_Length = 2  # cm
        self.Plywood_Length = 1  # cm
        self.Air_length = 8  # cm
        self.Drywood_Length = 1  # cm
        self.Brick_kValue = 0.77  # W/m*k
        self.Plywood_KValue = 0.1  # W/m*k
        self.Air_KValue = 0.024  # W/m*k
        self.Drywood_KValue = 0.6  # W/m*k
        self.Surface_Area_Wall = 297  # ft^2
        self.Surface_Area_Walls = 1188  # ft^2
        self.Height_Of_Wall = 9  # ft
        self.Length_Of_Floor = 33  # ft

    def DQDT_Conduction(self):
        return self.Surface_Area_Walls * (self.T_outside - self.T_inside) / \
               ((self.Brick_Length/self.Brick_kValue) + (self.Plywood_Length/self.Plywood_KValue) +
                (self.Air_length/self.Air_KValue) + (self.Drywood_Length/self.Drywood_KValue))


class Radiation:
    def __init__(self):
        self.sigma = 5.67 * 10 ^ 8
        self.T_sun = 5800  # k
        self.T_home = 300  # k
        self.A_sun = 6.07 * 10 ^ 18  # m2
        self.e_sun = 1

    def DQDT_Ratiation(self):
        return self.e_sun * self.sigma * self.A_sun * (pow(self.T_sun, 4) - pow(self.T_home, 4))


class Convection:
    def __init__(self):
        self.something = 8

    def DQDT_Convection(self):
        return




