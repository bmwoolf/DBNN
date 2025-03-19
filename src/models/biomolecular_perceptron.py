import numpy as np
from scipy.integrate import solve_ivp

class BiomolecularPerceptron:
    def __init__(self, u, v, gamma, phi):
        """
        Initializes the biomolecular perceptron parameters.
        :param u: Production rate of species Z1
        :param v: Production rate of species Z2
        :param gamma: Titration rate (sequestration strength)
        :param phi: Decay rate
        """
        self.u = u
        self.v = v
        self.gamma = gamma
        self.phi = phi
        
    def equations(self, t, z):
        z1, z2 = z
        dz1_dt = self.u - self.gamma * z1 * z2 - self.phi * z1
        dz2_dt = self.v - self.gamma * z1 * z2 - self.phi * z2
        return [dz1_dt, dz2_dt]
    
    def solve(self, z1_0=0, z2_0=0, t_span=(0, 10), t_eval=None):
        """
        Solves the system of ODEs over the given time span.
        :param z1_0: Initial condition for Z1
        :param z2_0: Initial condition for Z2
        :param t_span: Time span (start, end)
        :param t_eval: Optional list of times to evaluate the solution
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 100)

        sol = solve_ivp(self.equations, t_span, [z1_0, z2_0], t_eval=t_eval)
        return sol.t, sol.y
        
        