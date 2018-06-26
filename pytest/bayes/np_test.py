import numpy as np

test_vector = np.array([[4.4, 3.0, 1.3, 0.2, 1.0],
[4.4, 3.0, 1.3, 0.2, 1.0]])

# Create mask boolean array
mask = np.ones(len(test_vector), dtype=bool)

if test_vector.ndim <= 1:
  mask[test_vector.shape[0]-1] = False
else:
  # Create mask boolean array
  mask = np.ones(test_vector.shape, dtype=bool)
  mask[:, -1] = False

  test_vector = np.delete(test_vector, -1, 1)

data = test_vector[mask]

print(data.ndim)
print(data.ndim)
