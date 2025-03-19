import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from src.models.biomolecular_perceptron import BiomolecularPerceptron, BiomolecularNeuralNetwork 

def visualize_perceptron_dynamics(perceptron, z1_0, z2_0, ax, title):
    """Visualize the dynamics of a single perceptron"""
    t, sol = perceptron.solve(z1_0=z1_0, z2_0=z2_0)
    ax.plot(t, sol[0], label="Z1 (active)")
    ax.plot(t, sol[1], label="Z2 (sequestered)")
    ax.axhline(y=perceptron.threshold, color='r', linestyle='--', label='Threshold')
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title(title)
    ax.legend()

def visualize_network_dynamics(network, inputs):
    """Visualize the dynamics of the entire network"""
    n_layers = len(network.layers)
    max_perceptrons = max(len(layer) for layer in network.layers)
    
    # Create figure with subplots for each perceptron
    fig = plt.figure(figsize=(15, 5*n_layers))
    
    # Process each layer
    current_inputs = inputs
    for layer_idx, layer in enumerate(network.layers):
        layer_outputs = []
        
        # Plot each perceptron in the layer
        for p_idx, perceptron in enumerate(layer):
            ax = plt.subplot(n_layers, max_perceptrons, 
                           layer_idx * max_perceptrons + p_idx + 1)
            
            visualize_perceptron_dynamics(
                perceptron, 
                z1_0=current_inputs[0], 
                z2_0=current_inputs[1],
                ax=ax,
                title=f'Layer {layer_idx+1}, Perceptron {p_idx+1}\nInputs: {current_inputs}'
            )
            
            # Get output for next layer
            t, sol = perceptron.solve(z1_0=current_inputs[0], z2_0=current_inputs[1])
            output = perceptron.activation(sol[0][-1])
            layer_outputs.append(output)
        
        current_inputs = layer_outputs
    
    plt.tight_layout()
    return current_inputs  # Final output

def main():
    # Create network (same as in test)
    layer1 = [
        BiomolecularPerceptron(u=5, v=3, gamma=2, phi=0.5, threshold=1.0),
        BiomolecularPerceptron(u=4, v=2, gamma=1.5, phi=0.4, threshold=0.8)
    ]
    layer2 = [
        BiomolecularPerceptron(u=6, v=3, gamma=2, phi=0.5, threshold=1.2)
    ]
    network = BiomolecularNeuralNetwork(layers=[layer1, layer2])

    # Test different input combinations
    test_inputs = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ]

    # Create a figure for each input combination
    for inputs in test_inputs:
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Network Dynamics for Inputs: {inputs}', y=1.02, size=16)
        final_output = visualize_network_dynamics(network, inputs)
        print(f"Inputs: {inputs} -> Output: {final_output}")
        plt.show()

if __name__ == "__main__":
    main() 