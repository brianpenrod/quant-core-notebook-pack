import numpy as np

class PurgedTimeSeriesSplit:
    """
    Production implementation of Purged Cross-Validation.
    Prevents look-ahead bias and leakage in financial time series.
    """
    def __init__(self, n_splits=5, purge_gap=20):
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            test_start = (i + 1) * fold_size
            test_end = test_start + fold_size
            
            test_indices = indices[test_start:test_end]
            
            # Purge logic
            train_end = test_start - self.purge_gap
            
            if train_end < 1:
                continue 
                
            train_indices = indices[0:train_end]
            yield train_indices, test_indices
