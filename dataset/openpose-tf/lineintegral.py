import math
import numpy as np

# building the vectors
dx, dy = x2 — x1, y2 — y1
normVec = math.sqrt(dx ** 2 + dy ** 2)
vx, vy = dx/normVec, dy/normVec

# sampling
num_samples = 10
xs = np.arange(x1, x2, dx/num_samples).astype(np.int8)
ys = np.arange(y1, y2, dy/num_samples).astype(np.int8)

# evaluating on the field
pafXs = pafX[ys, xs]
pafYs = pafY[ys, xs]

# integral
score = sum(pafXs * vx + pafYs * vy) / num_samples
