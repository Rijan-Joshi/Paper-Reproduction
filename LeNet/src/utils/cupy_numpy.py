import os

USE_GPU = os.getenv("USE_ENV", "O") == "1"

if USE_GPU:
    try:
        import cupy as np
        print("Using GPU (CuPy)")
    except ImportError:
        import numpy as np
        print("CuPy not found - falling back to CPU")

else:
    import numpy as np
    print("Using CPU")

array = np.array
zeros = np.zeros
ones = np.ones
