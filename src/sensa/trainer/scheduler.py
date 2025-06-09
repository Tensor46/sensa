import math


def fn_cosine(start: float, end: float, iteration: int, iterations: int) -> float:
    """Compute a cosine-annealed interpolation between `start` and `end`.

    Args:
        start (float): Initial value at iteration 0.
        end (float): Final value at iteration `iterations - 1`.
        iteration (int): Current iteration index (0-based).
        iterations (int): Total number of iterations.

    Returns:
        float: Interpolated value.
               If `start == end` or `iteration >= iterations`, returns `end`.
    """
    if start == end or iteration >= iterations:  # value flats out
        return end
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * iteration / max(1, iterations - 1)))


def fn_linear(start: float, end: float, iteration: int, iterations: int) -> float:
    """Compute a linearly interpolated value between `start` and `end`.

    Args:
        start (float): Initial value at iteration 0.
        end (float): Final value at iteration `iterations - 1`.
        iteration (int): Current iteration index (0-based).
        iterations (int): Total number of iterations.

    Returns:
        float: Interpolated value.
               If `start == end` or `iteration >= iterations`, returns `end`.
    """
    if start == end or iteration >= iterations:  # value flats out
        return end
    return start + (end - start) * iteration / max(1, iterations - 1)
