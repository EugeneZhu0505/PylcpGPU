import cupy as cp

a = cp.random.randn(3, 100)
b = cp.array([1, 2, 4])[:, cp.newaxis]

a = a + b