import numpy as np

DTYPE_REAL = np.float64
DTYPE_UINT = np.uint32
DTYPE_INT  = np.int32
DTYPE_BOOL = np.bool_

EARTH_RADIUS = 6371.
WGS84_F  = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)