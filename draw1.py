import numpy as np
import matplotlib.pyplot as plt

# =======================
# 1) Задаём пример
# =======================
A = np.array([[1.0, 0.0],
              [0.0, 100.0]])
b = np.array([1.0, 1.0])

x0 = np.array([0.0, 0.0])
x_star = np.linalg.solve(A, b)

def f(x):
    return 0.5 * x @ (A @ x) - b @ x

# =======================
# 2) Градиентный спуск (наискорейший) с точным line search для квадратика
# r = b - A x  (это -градиент)
# alpha = (r^T r) / (r^T A r)
# =======================
def gradient_descent(A, b, x0, max_iter=100, tol=1e-12):
    x = x0.copy()
    xs = [x.copy()]
    r = b - A @ x
    bnorm = np.linalg.norm(b)

    for _ in range(max_iter):
        Ar = A @ r
        denom = r @ Ar
        if denom == 0:
            break
        alpha = (r @ r) / denom

        x = x + alpha * r
        xs.append(x.copy())

        r = b - A @ x
        if np.linalg.norm(r) / (bnorm + 1e-30) < tol:
            break

    return np.array(xs)

# =======================
# 3) Метод сопряжённых градиентов (CG) в стандартной форме
# r = b - A x
# p0 = r0
# alpha = (r^T r)/(p^T A p)
# beta  = (r_new^T r_new)/(r^T r)
# =======================
def conjugate_gradient(A, b, x0, max_iter=20, tol=1e-12):
    x = x0.copy()
    xs = [x.copy()]

    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    bnorm = np.linalg.norm(b)

    for _ in range(max_iter):
        Ap = A @ p
        denom = p @ Ap
        if denom == 0:
            break

        alpha = rs_old / denom
        x = x + alpha * p
        xs.append(x.copy())

        r = r - alpha * Ap
        rs_new = r @ r

        if np.sqrt(rs_new) / (bnorm + 1e-30) < tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return np.array(xs)

# Считаем траектории
xs_gd = gradient_descent(A, b, x0, max_iter=100)
xs_cg = conjugate_gradient(A, b, x0, max_iter=10)

print("Точное решение x* =", x_star)
print("Градиентный спуск: шагов =", len(xs_gd)-1, "последняя точка =", xs_gd[-1],
      "||b-Ax|| =", np.linalg.norm(b - A @ xs_gd[-1]))
print("Метод сопряжённых градиентов: шагов =", len(xs_cg)-1, "последняя точка =", xs_cg[-1],
      "||b-Ax|| =", np.linalg.norm(b - A @ xs_cg[-1]))

# =======================
# 4) Рисуем линии уровня "смотрибельно":
# растягиваем ось y: y2 = scale_y * y
# =======================
scale_y = 10.0  # растяжение оси y для визуализации

f_min = f(x_star)
deltas = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
levels = f_min + deltas
delta_max = deltas.max()

eigvals = np.linalg.eigvalsh(A)
lam_min, lam_max = eigvals.min(), eigvals.max()

rx = np.sqrt(2.0 * delta_max / lam_min)
ry = np.sqrt(2.0 * delta_max / lam_max)

pad_x = 0.25
pad_y = 0.06

x_min, x_max = x_star[0] - rx - pad_x, x_star[0] + rx + pad_x
y_min, y_max = x_star[1] - ry - pad_y, x_star[1] + ry + pad_y

nx, ny = 900, 700
x_grid = np.linspace(x_min, x_max, nx)
y2_grid = np.linspace(scale_y * y_min, scale_y * y_max, ny)

X, Y2 = np.meshgrid(x_grid, y2_grid)
Y = Y2 / scale_y

F = 0.5 * (X**2 + 100.0 * Y**2) - (X + Y)

# =======================
# 5) График + траектории (масштаб больше, маркеры тоньше, без прозрачности)
# =======================
fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

cs = ax.contour(X, Y2, F, levels=levels)
ax.clabel(cs, inline=True, fontsize=8)

# Точка минимума
ax.scatter([x_star[0]], [scale_y * x_star[1]],
           marker='x', s=120, linewidths=2, label="Точное решение $x^*$")

# Траектории: маркеры без заливки (видно фон), тонкая обводка, без alpha
ax.plot(xs_gd[:, 0], scale_y * xs_gd[:, 1],
        linestyle='-',
        marker='o', markersize=2.5,
        markerfacecolor='none',
        markeredgewidth=0.6,
        linewidth=0.9,
        markevery=1,   # <- маркер на каждом шаге
        label="Градиентный спуск (наискорейший)")

ax.plot(xs_cg[:, 0], scale_y * xs_cg[:, 1],
        linestyle='-',
        marker='s', markersize=4,
        markerfacecolor='none',
        markeredgewidth=0.8,
        linewidth=1.0,
        markevery=1,                 # CG шагов мало — можно все
        label="Метод сопряжённых градиентов")

ax.set_title(f"Линии уровня и траектории методов")
ax.set_xlabel("x")
ax.set_ylabel(f"{scale_y} · y")
ax.grid(True)
ax.legend()

# Немного "приблизим" область вокруг траекторий (масштаб)
# Можно подстроить коэффициенты, если хочешь ближе/дальше.
xmin = min(xs_gd[:, 0].min(), xs_cg[:, 0].min(), x_star[0]) - 0.05
xmax = max(xs_gd[:, 0].max(), xs_cg[:, 0].max(), x_star[0]) + 0.05
ymin = scale_y * (min(xs_gd[:, 1].min(), xs_cg[:, 1].min(), x_star[1]) - 0.02)
ymax = scale_y * (max(xs_gd[:, 1].max(), xs_cg[:, 1].max(), x_star[1]) + 0.02)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_aspect('equal', adjustable='box')
plt.show()