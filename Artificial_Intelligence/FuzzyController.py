import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# Fuzzy Controller
class FuzzyController:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    warmNorm = ctrl.Antecedent(np.arange(0, 1, 0.1), 'Warm Norm')
    hotNorm = ctrl.Antecedent(np.arange(0, 1, 0.1), 'Hot Norm')
    coolNorm = ctrl.Antecedent(np.arange(0, 1, 0.1), 'Cool Norm')
    coldNorm = ctrl.Antecedent(np.arange(0, 1, 0.1), 'Cold Norm')

    thermStat = ctrl.Consequent(np.arange(65, 76, 1), 'Set Thermostat')

    hotNorm.automf(3)
    warmNorm.automf(3)
    coolNorm.automf(3)
    coldNorm.automf(3)

    thermStat['Hot'] = fuzz.trimf(thermStat.universe, [65, 65, 70])
    thermStat['Warm'] = fuzz.trimf(thermStat.universe, [65, 68, 72])
    thermStat['Cool'] = fuzz.trimf(thermStat.universe, [68, 70, 75])
    thermStat['Cold'] = fuzz.trimf(thermStat.universe, [70, 75, 75])

    hotNorm.view()
    warmNorm.view()
    coolNorm.view()
    coldNorm.view()
    thermStat.view()

    rule1 = ctrl.Rule(hotNorm['good'] & warmNorm['average'] & coolNorm['poor'] & coldNorm['poor'], thermStat['Hot'])
    rule2 = ctrl.Rule(hotNorm['average'] & warmNorm['good'] & coolNorm['average'] & coldNorm['poor'],
                      thermStat['Warm'])
    rule3 = ctrl.Rule(hotNorm['poor'] & warmNorm['average'] & coolNorm['good'] & coldNorm['average'],
                      thermStat['Cool'])
    rule4 = ctrl.Rule(hotNorm['poor'] & warmNorm['poor'] & coolNorm['average'] & coldNorm['good'],
                      thermStat['Cold'])
    rule5 = ctrl.Rule(hotNorm['poor'] & warmNorm['poor'] & coolNorm['poor'] & coldNorm['poor'], thermStat['Hot'])
    rule6 = ctrl.Rule(hotNorm['poor'] & warmNorm['poor'] & coolNorm['poor'] & coldNorm['good'], thermStat['Cold'])
    rule7 = ctrl.Rule(hotNorm['poor'] & warmNorm['poor'] & coolNorm['good'] & coldNorm['poor'], thermStat['Cool'])
    rule8 = ctrl.Rule(hotNorm['poor'] & warmNorm['good'] & coolNorm['poor'] & coldNorm['poor'], thermStat['Warm'])
    rule9 = ctrl.Rule(hotNorm['good'] & warmNorm['poor'] & coolNorm['poor'] & coldNorm['poor'], thermStat['Hot'])

    thermStat_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    thermStat_set = ctrl.ControlSystemSimulation(thermStat_ctrl)

    def inference(self):
        temperatures = []
        for i in range(len(self.probabilities)):
            self.thermStat_set.input['Cold Norm'] = self.probabilities[i][0].detach().numpy()
            self.thermStat_set.input['Cool Norm'] = self.probabilities[i][1].detach().numpy()
            self.thermStat_set.input['Warm Norm'] = self.probabilities[i][2].detach().numpy()
            self.thermStat_set.input['Hot Norm'] = self.probabilities[i][3].detach().numpy()

            self.thermStat_set.compute()
            temperatures.append(self.thermStat_set.output['Set Thermostat'])
            # print(thermStat_set.output['Set Thermostat'])
        return temperatures
