import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
from src.models.biomolecular_perceptron import BiomolecularPerceptron 

# Define the parameters
model = BiomolecularPerceptron(u=5, v=3, gamma=2, phi=0.5)

# Solve the system
t, sol = model.solve()

# Plot results
plt.plot(t, sol[0], label="Z1 (active species)")
plt.plot(t, sol[1], label="Z2 (sequestered species)")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.show() 