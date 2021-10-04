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



class Convection:
    def __init__(self):
        self.something = 8

    def DQDT_Convection(self):
        return