import numpy as np

def generate_distribution(dist_type='normal', size=100, **kwargs):
    """
    Generate synthetic data from various mathematical distributions.

    Parameters:
    - dist_type: str, distribution type ('normal', 'uniform', 'exponential', 'poisson', 'binomial')
    - size: int, number of samples
    - kwargs: distribution-specific parameters

    Returns:
    - numpy array of generated data
    """
    if dist_type == 'normal':
        return np.random.normal(loc=kwargs.get('mean', 0),
                                scale=kwargs.get('std', 1),
                                size=size)
    elif dist_type == 'uniform':
        return np.random.uniform(low=kwargs.get('low', 0),
                                 high=kwargs.get('high', 1),
                                 size=size)
    elif dist_type == 'exponential':
        return np.random.exponential(scale=kwargs.get('scale', 1), size=size)
    elif dist_type == 'poisson':
        return np.random.poisson(lam=kwargs.get('lam', 1), size=size)
    elif dist_type == 'binomial':
        return np.random.binomial(n=kwargs.get('n', 1), 
                                  p=kwargs.get('p', 0.5),
                                  size=size)
    else:
        raise ValueError(f"Distribution {dist_type} not supported.")
