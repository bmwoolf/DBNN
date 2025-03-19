import numpy as np
from scipy.integrate import solve_ivp

class BiomolecularPerceptron:
    def __init__(self, u, v, gamma, phi, threshold=0):
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
        self.threshold = threshold
    
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
        
    def activation(self, z1_final):
        """
        Applies a ReLU-like activation function to determine the output of the perceptron.
        :param z1_final: Final concentration of Z1
        :return: 1 if z1_final is above threshold, 0 otherwise
        """
        return 1 if z1_final >= self.threshold else 0

class BiomolecularNeuralNetwork:
    def __init__(self, layers):
        """
        Initializes the multilayer biomolecular neural network with a list of layers.
        :param layers: List of layers, each containing a list of BiomolecularPerceptron instances
        """
        self.layers = layers
    
    def forward(self, inputs):
        """
        Processes inputs through the network layer by layer.
        :param inputs: List of input values to the first layer 
        :return: Final layer outputs
        """
        layer_inputs = inputs 
        for layer in self.layers:
            layer_outputs = []
            for perceptron in layer:
                t, sol = perceptron.solve(z1_0=layer_inputs[0], z2_0=layer_inputs[1])
                output = perceptron.activation(sol[0][-1])
                layer_outputs.append(output)
            # Pass outputs as inputs to next layer
            layer_inputs = layer_outputs
        return layer_outputs