import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pytest
import numpy as np
from src.models.biomolecular_perceptron import BiomolecularPerceptron, BiomolecularNeuralNetwork

# Move parametrized test outside the class
@pytest.mark.parametrize("layer_sizes", [
    ([1, 1]),      # Minimal network
    ([2, 1]),      # Current test network
    ([3, 2, 1]),   # Deeper network
    ([4, 3, 2, 1]) # Even deeper network
])
def test_different_architectures(layer_sizes):
    """Test different network architectures"""
    # Create layers based on sizes
    layers = []
    for size in layer_sizes:
        # Create a layer with 'size' number of perceptrons
        layer = []
        for _ in range(size):
            perceptron = BiomolecularPerceptron(u=5, v=3, gamma=2, phi=0.5, threshold=1.0)
            layer.append(perceptron)
        layers.append(layer)
    
    network = BiomolecularNeuralNetwork(layers=layers)
    inputs = [1.0, 1.0]
    output = network.forward(inputs)
    
    # Check output shape matches final layer
    assert len(output) == layer_sizes[-1]
    # Check binary outputs
    for out in output:
        assert out in [0, 1]

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

    def test_multilayer_propagation(self):
        layer1 = [
            BiomolecularPerceptron(u=5, v=3, gamma=2, phi=0.5, threshold=1.5),
            BiomolecularPerceptron(u=4, v=2, gamma=1.5, phi=0.4, threshold=1.2)
        ]
        layer2 = [BiomolecularPerceptron(u=6, v=3, gamma=2, phi=0.5, threshold=1.8)]
        network = BiomolecularNeuralNetwork(layers=[layer1, layer2])

        inputs = [1.0, 1.5]
        output = network.forward(inputs)
        self.assertIsInstance(output, list)

    def test_network_stability(self):
        """Test if network reaches stable state"""
        inputs = [1.0, 1.0]
        output1 = self.network.forward(inputs)
        output2 = self.network.forward(inputs)
        self.assertEqual(output1, output2, "Network should give consistent outputs")

    def test_edge_cases(self):
        """Test network behavior with edge cases"""
        test_cases = [
            ([0.0, 0.0], "zero"),
            ([1e6, 1e6], "very large"),
            ([-1.0, -1.0], "negative"),
            ([1e-6, 1e-6], "very small")
        ]
        
        for inputs, case in test_cases:
            try:
                output = self.network.forward(inputs)
                for out in output:
                    self.assertIn(out, [0, 1], f"Output for {case} inputs not binary")
            except Exception as e:
                self.fail(f"Network failed for {case} inputs: {str(e)}")

if __name__ == '__main__':
    unittest.main() 