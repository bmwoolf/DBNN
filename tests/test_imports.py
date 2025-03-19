# Try importing each package
try:
    import numpy
    print("numpy version:", numpy.__version__)
except ImportError as e:
    print("Failed to import numpy:", e)

try:
    import scipy
    print("scipy version:", scipy.__version__)
except ImportError as e:
    print("Failed to import scipy:", e)

try:
    import matplotlib
    print("matplotlib version:", matplotlib.__version__)
except ImportError as e:
    print("Failed to import matplotlib:", e) 