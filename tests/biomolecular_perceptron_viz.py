from src.models.biomolecular_perceptron import BiomolecularPerceptron
import matplotlib.pyplot as plt

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