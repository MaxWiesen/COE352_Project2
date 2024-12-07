import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg


def get_elemental_mass(h):
    # Form 2D per-element phi matrix by element map
    a = 1
    b = 0
    # Behind node, integral from 0 to 1 of (1-ξ)^2
    back = lambda h: h * (-np.power(1 - a, 3) / 3 - -np.power(1 - b, 3) / 3)
    # On node, integral from 0 to 1 of (1-ξ)ξ
    on = lambda h: h * (-a * np.power(1 - a, 2) / 2 - -b * np.power(1 - b, 2) / 2) + 1/2 * back(h)
    # Ahead of node, integral from 0 to 1 of ξ^2
    forward = lambda h: h * (np.power(a, 3) / 3 - np.power(b, 3) / 3)

    return np.array([[back(h), on(h)], [on(h), forward(h)]])


def get_elemental_stiffness(h):
    # Form 2D per-element phi-prime matrix by element map
    a = 1
    b = 0
    # Behind node, integral from 0 to 1 of (-1/h)^2
    back = lambda h: 1 / h
    # On node, integral from 0 to 1 of (-1/h)(1/h)
    on = lambda h: -1 / h
    # Ahead of node, integral from 0 to 1 of (1/h)^2
    forward = back

    return np.array([[back(h), on(h)], [on(h), forward(h)]])


def create_matrices(grid):
    h = grid[1] - grid[0]

    M = np.zeros((len(grid), len(grid)))
    Me = get_elemental_mass(h)

    K = np.zeros((len(grid), len(grid)))
    Ke = get_elemental_stiffness(h)

    for i in range(len(grid) - 1):
        M[i:i+2, i:i+2] += Me
        K[i:i+2, i:i+2] += Ke

    return M, K


def forward_euler(f, xs, u0, dirichlet, M_inv, K, dt):
    u = [u0]
    t = dt
    h = xs[1] - xs[0]
    for i in range(1, int(1/dt)):
        # Gaussian Quadrature
        # Map: x = xe + hξ => integral from 0 to 1 of f(ξ, t) * phi_hat(ξ) * h
        original = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        xis = (1 + original) / 2
        w = np.array([1, 1]) / 2
        F = w[0] * h * f(t)(xs + h * xis[0]) * (1-xis[0])
        F += w[1] * h * f(t)(xs + h * xis[1]) * xis[1]

        # Forward Euler implementation
        u.append(u[i-1] + (F - K @ u[i-1]) @ M_inv * dt)

        # Reinforce bounds
        u[-1][0] = dirichlet[0]
        u[-1][-1] = dirichlet[1]

        t += dt

    return np.array(u)


def backward_euler(f, xs, u0, dirichlet, M, K, dt):
    u = [u0]
    t = dt
    h = xs[1] - xs[0]
    for i in range(1, int(1/dt)):
        # Gaussian Quadrature
        # Map: x = xe + hξ => integral from 0 to 1 of f(ξ, t) * phi_hat(ξ) * h
        original = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        xis = (1 + original) / 2
        w = np.array([1, 1]) / 2
        F = w[0] * h * f(t + dt)(xs + h * xis[0]) * (1-xis[0])
        F += w[1] * h * f(t + dt)(xs + h * xis[1]) * xis[1]

        # Backward Euler implementation
        u.append(np.linalg.inv(M / dt + K) @ (F + M @ u[i-1] / dt))

        # Reinforce bounds
        u[-1][0] = dirichlet[0]
        u[-1][-1] = dirichlet[1]

        t += dt

    return np.array(u)


def plot_3d(xs, u, dt, f):
    t = np.arange(0, 1, dt)
    X, T = np.meshgrid(xs, t)

    X_flat = X.flatten()
    T_flat = T.flatten()
    u_flat = u.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_flat, T_flat, u_flat, c=u_flat, cmap='viridis', s=50)
    ax.set_xlabel('X')
    ax.set_ylabel('Time')
    ax.set_zlabel('u')
    ax.set_title(f'U in Space Over Time [dt=1/{f}]')
    plt.show()


def plot_end_state(xs, u_end, f, analytic=None):
    fig, ax = plt.subplots()
    if analytic:
        ax.plot(xs, u_end, label='FE Approximation')
        ax.plot(xs, analytic(xs), 'r--', label='Analytical Solution')
        fig.legend()
    else:
        ax.plot(xs, u_end)
    ax.grid()
    ax.set_title(f'FEM End State [dt=1/{f}]')
    ax.set_xlabel('X')
    ax.set_ylabel('u')
    plt.show()


def run_fem(xs, dt, dirichlet, forward=True):
    M, K = create_matrices(xs)
    if forward:
        M_inv = np.linalg.inv(M)
        return forward_euler(lambda t: lambda x: (np.pi*np.pi - 1) * np.exp(-t) * np.sin(np.pi * x), xs, np.sin(np.pi * xs), dirichlet, M_inv, K, dt)
    else:
        return backward_euler(lambda t: lambda x: (np.pi*np.pi - 1) * np.exp(-t) * np.sin(np.pi * x), xs, np.sin(np.pi * xs), dirichlet, M, K, dt)


def main():
    N = 11
    xs = np.linspace(0, 1, N)
    f = 5
    dt = 1 / f
    dirichlet = [0, 0]
    forward_euler = False

    u = run_fem(xs, dt, dirichlet, forward_euler)

    plot_3d(xs, u, dt, f)
    plot_end_state(xs, u[-1], f, lambda x: np.exp(-1) * np.sin(np.pi * x))


if __name__ == '__main__':
    main()
