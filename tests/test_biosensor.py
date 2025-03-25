# because we cannot simulate CAR T cells or bioreactors (Pioreactor still hasn't shipped), we will test with a simulated biosensor

"""
Inputs:
	- Biomarker A (PSA for prostate cancer)
	- Biomarker B (HER2 for breast cancer)
	- Biomarker C (inflammatory cytokine IL-6)

BNN Layers:
	- Layer 1: Processes raw biomarker concentrations
	- Layer 2: Detects complex patterns (e.g., elevated PSA + IL-6 = cancer)
	- Output: Biosensor turns "ON" (1) if cancer is detected, otherwise "OFF" (0)
"""

import pytest
import numpy as np
from src.models.biomolecular_perceptron import BiomolecularPerceptron, BiomolecularNeuralNetwork

def create_biosensor_network():
    """Creates a BNN configured for biosensor detection"""
    # Layer 1: Process individual biomarkers with clinically relevant thresholds
    layer1 = [
        BiomolecularPerceptron(u=15, v=3, gamma=1.0, phi=0.3, threshold=1.8),  # PSA detector (>2.0 ng/mL suspicious)
        BiomolecularPerceptron(u=15, v=3, gamma=1.0, phi=0.3, threshold=1.8),  # HER2 detector (>2.0 overexpression)
        BiomolecularPerceptron(u=15, v=3, gamma=1.0, phi=0.3, threshold=1.8),  # IL-6 detector (>2.0 pg/mL inflammatory)
    ]
    
    # Layer 2: Pattern detection - activates if any biomarker is significantly elevated
    layer2 = [
        BiomolecularPerceptron(u=12, v=3, gamma=1.0, phi=0.3, threshold=0.5)  # Final classifier
    ]
    
    return BiomolecularNeuralNetwork(layers=[layer1, layer2])

@pytest.mark.parametrize("biomarkers, expected", [
    ([0.0, 0.0], 0),  # Healthy - low levels of all markers
    ([5.0, 0.0], 1),  # High PSA only - potential prostate cancer
    ([0.0, 4.0], 1),  # High HER2 only - potential breast cancer
    ([3.0, 3.0], 1),  # Multiple elevated markers - likely cancer
])
def test_cancer_detection(biomarkers, expected):
    """Test if biosensor correctly identifies cancer markers"""
    network = create_biosensor_network()
    result = network.classify_biosensor(biomarkers)
    assert result == expected, f"Expected {expected} for biomarkers {biomarkers}, got {result}"

def test_biosensor_threshold_sensitivity():
    """Test if biosensor responds appropriately to different concentration thresholds"""
    network = create_biosensor_network()
    
    # Test with concentrations just below thresholds
    low_markers = [1.0, 1.0]  # Just below detection thresholds
    assert network.classify_biosensor(low_markers) == 0, "Should not detect at sub-threshold levels"
    
    # Test with concentrations just above thresholds
    high_markers = [3.0, 3.0]  # Just above detection thresholds
    assert network.classify_biosensor(high_markers) == 1, "Should detect at above-threshold levels"

def test_biosensor_noise_robustness():
    """Test if biosensor is robust to small noise in measurements"""
    network = create_biosensor_network()
    base_markers = [3.0, 3.0]
    
    # Add small random noise
    np.random.seed(42)
    noisy_results = []
    for _ in range(10):
        noise = np.random.normal(0, 0.1, size=2)
        noisy_markers = [max(0, m + n) for m, n in zip(base_markers, noise)]
        noisy_results.append(network.classify_biosensor(noisy_markers))
    
    # All results should be consistent despite noise
    assert all(r == noisy_results[0] for r in noisy_results), "Biosensor should be robust to small noise"

def test_time_series_response():
    """Test biosensor response over time"""
    network = create_biosensor_network()
    
    # Simulate increasing biomarker concentrations
    time_points = np.linspace(0, 5, 10)
    responses = []
    
    for t in time_points:
        # Simulate gradual increase in biomarkers
        markers = [t, t]
        responses.append(network.classify_biosensor(markers))
    
    # Should transition from 0 to 1 as concentrations increase
    assert responses[0] == 0, "Should start negative"
    assert responses[-1] == 1, "Should end positive"
    assert sum(responses) > 0, "Should detect disease at some point"