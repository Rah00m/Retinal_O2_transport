"""
Reference O2 pressure data (confirmed by the user), separate from the
verified base-case book parameters in config.py.

Note: REF_IR[0]=26 / REF_CC[-1]=130 differ from the base case's
P_IR_S=20 / P_CC_S=100, so this data reflects a different scenario
than the base case solved in physics.py/config.py. It's used only in
the optional data-assisted experiment (evaluate_model.py
--with-reference-data), not as ground truth for the physics-only run.
"""
import numpy as np

REF_IR = np.array([26.000, 25.714, 25.519, 25.428, 25.441, 25.558, 25.779, 26.104, 26.533, 27.066, 27.703])
REF_OR = np.array([27.716, 28.457, 29.315, 30.303, 31.408, 32.630, 33.982, 35.477, 37.115, 38.909, 40.846])
REF_FL = np.array([40.846, 42.939, 45.214, 47.671, 50.310, 53.157, 56.212, 59.501, 63.024, 66.794, 70.824])
REF_CC = np.array([70.837, 75.140, 79.755, 84.695, 89.960, 95.589, 101.608, 108.030, 114.881, 122.200, 130.000])

REF_BY_LAYER = {"IR": REF_IR, "OR": REF_OR, "FL": REF_FL, "CC": REF_CC}

MISMATCH_WARNING = (
    "Note: REF data (P={:.1f}\u2192{:.1f}) is a different scenario than "
    "the base case (P=20\u2192100) used for physics-only training."
).format(REF_IR[0], REF_CC[-1])