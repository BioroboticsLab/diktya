import multiprocessing
import numpy as np
import pprint
from itertools import repeat


def _value_and_space(t):
    f, x = t
    return (f(x), x)


def _print_best(iteration, val, old_best, space):
    pp = pprint.PrettyPrinter(indent=4)
    best_val = val
    improvement = 100 - 100*best_val / old_best
    print("[{}] New best: {}, Improvement: {:.2f}%"
          .format(iteration, best_val, improvement))
    print("Space is:", end='')
    pp.pprint(space)
    print()


def fmin(f, space_func, n=50, n_jobs='n_cpus', verbose=0):
    """
    Minimizes ``f`` by using random search.

    Args:
        f (function): function to optimize. Gets output of space_func as input.
        space_func (function): Returns random samples form the search space.
        n (int): Number of samples to run. Default 50
        n_jobs (int|str): Number of parallel jobs. Use ``'n_cpus'` for
            same amount as cpus avialable. Default ``'n_cpus'``.

    Simple Example:

    ::

        def quadratic_function():
            return (x - 2) ** 2

        def space_function():
            return np.random.uniform(-2, 4)

        results = fmin(quadratic_function, space_function, n=50)
        # sorted by score
        print("Min score: {}".format(results[0][0]))

    """
    if n_jobs == 'n_cpus':
        n_jobs = multiprocessing.cpu_count()

    with multiprocessing.Pool(n_jobs) as pool:
        results = []
        best_val = np.inf
        generator = zip(repeat(f), [space_func() for _ in range(n)])
        for i, (val, space) in enumerate(pool.imap(_value_and_space,
                                                   generator)):
            if val < best_val:
                old_best = best_val
                best_val = val
                if verbose > 0:
                    _print_best(i, best_val, old_best, space)
            results.append((val, space))

    results.sort(key=lambda t: t[0])
    return results
