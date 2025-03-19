import unittest
import numpy as np
from src.models.biomolecular_perceptron import BiomolecularPerceptron, BiomolecularNeuralNetwork

class TestBiomolecularNeuralNetwork(unittest.TestCase):
    def setUp(self):
        # Create a simple 2-layer network (2 perceptrons in first layer, 1 in second)
        self.layer1 = [
            BiomolecularPerceptron(u=5, v=3, gamma=2, phi=0.5, threshold=1.0),
            BiomolecularPerceptron(u=4, v=2, gamma=1.5, phi=0.4, threshold=0.8)
        ]
        self.layer2 = [
            BiomolecularPerceptron(u=6, v=3, gamma=2, phi=0.5, threshold=1.2)
        ]
        self.network = BiomolecularNeuralNetwork(layers=[self.layer1, self.layer2])

    def test_network_initialization(self):
        # Test network structure
        self.assertEqual(len(self.network.layers), 2)
        self.assertEqual(len(self.network.layers[0]), 2)  # First layer has 2 perceptrons
        self.assertEqual(len(self.network.layers[1]), 1)  # Second layer has 1 perceptron

    def test_forward_pass(self):
        # Test forward pass with sample inputs
        inputs = [1.0, 1.5]  # Initial concentrations
        output = self.network.forward(inputs)
        
        # Check output type and shape
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), 1)  # Single output from final layer
        self.assertIn(output[0], [0, 1])  # Binary output from activation

    def test_layer_connectivity(self):
        # Test that each layer's output can be used as input for the next layer
        inputs = [0.5, 0.8]
        
        # First layer processing
        layer1_outputs = []
        for perceptron in self.layer1:
            t, sol = perceptron.solve(z1_0=inputs[0], z2_0=inputs[1])
            output = perceptron.activation(sol[0][-1])
            layer1_outputs.append(output)
        
        # Verify layer1 outputs
        self.assertEqual(len(layer1_outputs), 2)
        for output in layer1_outputs:
            self.assertIn(output, [0, 1])
        
        # Second layer processing
        layer2_outputs = []
        for perceptron in self.layer2:
            t, sol = perceptron.solve(z1_0=layer1_outputs[0], z2_0=layer1_outputs[1])
            output = perceptron.activation(sol[0][-1])
            layer2_outputs.append(output)
        
        # Verify layer2 outputs
        self.assertEqual(len(layer2_outputs), 1)
        self.assertIn(layer2_outputs[0], [0, 1])

    def test_different_inputs(self):
        # Test network with different input combinations
        test_inputs = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ]
        
        for inputs in test_inputs:
            output = self.network.forward(inputs)
            self.assertEqual(len(output), 1)
            self.assertIn(output[0], [0, 1])

if __name__ == '__main__':
    unittest.main() 