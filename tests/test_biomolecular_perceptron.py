import unittest
import numpy as np
from src.models.biomolecular_perceptron import BiomolecularPerceptron

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

if __name__ == '__main__':
    unittest.main() 