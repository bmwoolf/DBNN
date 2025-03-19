import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pytest
import numpy as np
from src.models.biomolecular_perceptron import BiomolecularPerceptron

# Move parametrized test outside the class
@pytest.mark.parametrize("u, v, gamma, phi, threshold, expected", [
    (0, 0, 1, 1, 0.5, 0),          # No production → No activation
    (100, 100, 1, 1, 0.5, 1),      # High production → Always activates
    (2, 3, 20, 0.5, 5.0, 0),       # Low production + high threshold → Should not activate
    (10, 3, 0.1, 0.5, 1.5, 1),     # High production + low sequestration → Should activate
])
def test_extreme_cases(u, v, gamma, phi, threshold, expected):
    perceptron = BiomolecularPerceptron(u, v, gamma, phi, threshold)
    t, sol = perceptron.solve()
    output = perceptron.activation(sol[0][-1])
    assert output == expected, f"Expected {expected}, got {output} (Z1={sol[0][-1]})"

class TestBiomolecularPerceptron(unittest.TestCase):
    def setUp(self):
        self.model = BiomolecularPerceptron(u=5, v=3, gamma=2, phi=0.5)
    
    def test_initialization(self):
        self.assertEqual(self.model.u, 5)
        self.assertEqual(self.model.v, 3)
        self.assertEqual(self.model.gamma, 2)
        self.assertEqual(self.model.phi, 0.5)
    
    def test_solve(self):
        t, sol = self.model.solve(t_span=(0, 1))
        # Check shapes
        self.assertEqual(len(t), 100)  # Default points
        self.assertEqual(sol.shape, (2, 100))  # 2 species, 100 timepoints
        
        # Check initial conditions
        np.testing.assert_almost_equal(sol[0][0], 0)  # Z1 starts at 0
        np.testing.assert_almost_equal(sol[1][0], 0)  # Z2 starts at 0)

    def test_basic_perceptron(self):
        perceptron = BiomolecularPerceptron(u=5, v=3, gamma=2, phi=0.5, threshold=1.5)
        t, sol = perceptron.solve()
        z1_final = sol[0][-1]
        assert perceptron.activation(z1_final) in [0, 1], "Output should be 0 or 1"

    def test_threshold_behavior(self):
        # Adjust parameters to ensure it's below threshold
        perceptron = BiomolecularPerceptron(u=2, v=3, gamma=2, phi=0.5, threshold=3.0)
        
        # Case 1: Below threshold
        t, sol = perceptron.solve()
        self.assertEqual(perceptron.activation(sol[0][-1]), 0, 
                        "Should not activate below threshold")

        # Case 2: Above threshold
        perceptron = BiomolecularPerceptron(u=10, v=3, gamma=2, phi=0.5, threshold=1.5)
        t, sol = perceptron.solve()
        self.assertEqual(perceptron.activation(sol[0][-1]), 1, 
                        "Should activate above threshold")

    def test_convergence_and_stability(self):
        """Test if perceptron reaches stable state"""
        perceptron = BiomolecularPerceptron(u=5, v=3, gamma=2, phi=0.5, threshold=1.5)
        t, sol = perceptron.solve(t_span=(0, 50))  # Longer time to ensure convergence
        
        # Check if final values are stable
        last_points = sol[:, -10:]  # Last 10 timepoints
        for species in last_points:
            std = np.std(species)
            assert std < 0.01, f"Species concentration not stable: std={std}"

    def test_reproducibility(self):
        """Test if same parameters give same results"""
        perceptron1 = BiomolecularPerceptron(u=5, v=3, gamma=2, phi=0.5, threshold=1.5)
        perceptron2 = BiomolecularPerceptron(u=5, v=3, gamma=2, phi=0.5, threshold=1.5)
        
        t1, sol1 = perceptron1.solve()
        t2, sol2 = perceptron2.solve()
        
        np.testing.assert_array_almost_equal(sol1, sol2)

if __name__ == '__main__':
    unittest.main() 