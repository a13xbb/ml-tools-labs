import numpy as np

class DataLoader:
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        self.indices = np.arange(self.num_samples)
        
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.cur_idx = 0
        return self
    
    def __next__(self):
        if self.cur_idx >= self.num_samples:
            raise StopIteration
        
        start_idx = self.cur_idx
        end_idx = min(self.num_samples, start_idx + self.batch_size)
        batch_indices = self.indices[start_idx: end_idx]
        
        self.cur_idx = end_idx
        
        return self.X[batch_indices], self.y[batch_indices]