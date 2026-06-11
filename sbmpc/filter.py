import jax.numpy as jnp


def cubic_spline(x, y, x_new):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    x_new = jnp.asarray(x_new)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if y.ndim < 1:
        raise ValueError("y must have at least one dimension")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must share the same leading dimension")
    if x.shape[0] < 2:
        raise ValueError("at least two control points are required")

    n = x.shape[0]
    tail_shape = y.shape[1:]
    y_flat = y.reshape(n, -1)

    if n == 2:
        span = x[1] - x[0]
        alpha = ((x_new - x[0]) / span).reshape(-1, 1)
        y_new = y_flat[0] + alpha * (y_flat[1] - y_flat[0])
        return y_new.reshape(x_new.shape + tail_shape)

    h = x[1:] - x[:-1]
    if jnp.any(h <= 0):
        raise ValueError("x must be strictly increasing")

    slopes = (y_flat[1:] - y_flat[:-1]) / h[:, None]

    diag = jnp.concatenate(
        [
            jnp.array([1.0], dtype=y.dtype),
            2.0 * (h[:-1] + h[1:]),
            jnp.array([1.0], dtype=y.dtype),
        ]
    )
    lower = jnp.concatenate([h[:-1], jnp.zeros(1, dtype=y.dtype)])
    upper = jnp.concatenate([jnp.zeros(1, dtype=y.dtype), h[1:]])

    system = jnp.diag(diag)
    system = system.at[jnp.arange(1, n), jnp.arange(0, n - 1)].set(lower)
    system = system.at[jnp.arange(0, n - 1), jnp.arange(1, n)].set(upper)

    rhs = jnp.zeros((n, y_flat.shape[1]), dtype=y.dtype)
    rhs = rhs.at[1:-1].set(3.0 * (slopes[1:] - slopes[:-1]))

    c = jnp.linalg.solve(system, rhs)
    b = slopes - h[:, None] * (2.0 * c[:-1] + c[1:]) / 3.0
    d = (c[1:] - c[:-1]) / (3.0 * h[:, None])
    a = y_flat[:-1]

    interval_idx = jnp.clip(
        jnp.searchsorted(x[1:], x_new, side="right"),
        0,
        n - 2,
    )
    dx = (x_new - x[interval_idx]).reshape(-1, 1)

    y_new = (
        a[interval_idx]
        + b[interval_idx] * dx
        + c[interval_idx] * dx**2
        + d[interval_idx] * dx**3
    )
    return y_new.reshape(x_new.shape + tail_shape)


def cubic_spline_matrix(x, x_new):
    basis = jnp.eye(jnp.asarray(x).shape[0], dtype=jnp.float32)
    return cubic_spline(x, basis, x_new)
