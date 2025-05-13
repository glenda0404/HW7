import numpy as np

def jacobi(A, b, x0, tol=1e-10, max_iter=1000):
    D = np.diag(np.diag(A))
    R = A - D
    x = x0.copy()
    for _ in range(max_iter):
        x_new = np.linalg.inv(D) @ (b - R @ x)
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def gauss_seidel(A, b, x0, tol=1e-10, max_iter=1000):
    x = x0.copy()
    n = len(x)
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def sor(A, b, x0, omega=1.25, tol=1e-10, max_iter=1000):
    x = x0.copy()
    n = len(x)
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = x[i] + omega * ((b[i] - s1 - s2) / A[i][i] - x[i])
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def conjugate_gradient(A, b, x0, tol=1e-10, max_iter=1000):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    for _ in range(max_iter):
        Ap = A @ p
        alpha = r @ r / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) < tol:
            return x
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
    return x

# 題目矩陣
A = np.array([
    [4, -1,  0, -1,  0,  0],
    [-1, 4, -1,  0, -1,  0],
    [0, -1,  4,  0,  1, -1],
    [-1, 0,  0,  4, -1, -1],
    [0, -1,  0, -1,  4, -1],
    [0,  0, -1,  0, -1,  4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)
x0 = np.zeros_like(b)

# 執行各方法
x_jacobi = jacobi(A, b, x0)
x_gs = gauss_seidel(A, b, x0)
x_sor = sor(A, b, x0, omega=1.25)
x_cg = conjugate_gradient(A, b, x0)

# 顯示結果
print("=== Iterative Solutions ===")
print("Jacobi:           ", x_jacobi)
print("Gauss-Seidel:     ", x_gs)
print("SOR (ω=1.25):     ", x_sor)
print("Conjugate Gradient:", x_cg)
