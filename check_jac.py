import numpy as np

def f(u, X, B, beta):
    v = B @ u
    y = X * np.exp(-v)
    return (1/beta) * np.diag(u) @ B.T @ y

def analytical_jacobian(u, X, B, beta):
    v = B @ u
    y = X * np.exp(-v)
    diag_y = np.diag(y)
    diag_u = np.diag(u)
    term1 = np.diag(B.T @ y)
    term2 = diag_u @ B.T @ diag_y @ B
    return (1/beta) * (term1 - term2)

def numerical_jacobian(f, u, X, B, beta, epsilon=1e-5):
    n = u.shape[0]
    m = f(u, X, B, beta).shape[0]
    J = np.zeros((m, n))
    for i in range(n):
        u_plus = np.copy(u)
        u_minus = np.copy(u)
        u_plus[i] += epsilon
        u_minus[i] -= epsilon
        f_plus = f(u_plus, X, B, beta)
        f_minus = f(u_minus, X, B, beta)
        J[:, i] = (f_plus - f_minus) / (2 * epsilon)
    return J

# Randomly generated variables
n = 4  # Size of u
m = 5  # Size of X and rows of B
u = np.random.rand(n)
X = np.random.rand(m)
B = np.random.rand(m, n)
beta = 0.5  # A constant between 0 and 1

# Calculate the analytical Jacobian
J_analytical = analytical_jacobian(u, X, B, beta)

# Calculate the numerical Jacobian
J_numerical = numerical_jacobian(f, u, X, B, beta)

# Compare the Jacobians
print("Analytical Jacobian:\n", J_analytical)
print("Numerical Jacobian:\n", J_numerical)
print("Difference:\n", J_analytical - J_numerical)
print("difference:", np.linalg.norm(J_analytical - J_numerical))
