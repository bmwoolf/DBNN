![DBNN Banner](assets/github_banner.png)

# DBNN
A BNN simulation framework to test biomolecular neural circuits before lab implementation. 

This is based on the paper [A Dynamical Biomolecular Neural Network](https://www.researchgate.net/publication/344958092_A_Dynamical_Biomolecular_Neural_Network) from the Weiss Lab at MIT.

This can be used by synthetic biologists designing genetic circuits, AI/ML researchers applying computational methods to biology, and biotech companies that are optimizing gene therapy and biosensors. 

The example test cases rely on CAR T cell logic gates. 

## Installation

```bash
python3.12 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .  # Install package in development mode for testing
```

## Testing

```bash
# Run unit tests
pytest tests/test_biomolecular_perceptron.py
pytest tests/test_biomolecular_neural_network.py
```

## Visualization

```bash
# Run visualization scripts
python tests/biomolecular_perceptron_viz.py
python tests/biomolecular_neural_network_viz.py
```
