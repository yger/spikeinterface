import numpy as np

class UniformSelection:
    
    name = "uniform"
    need_noise_levels = False
    params_doc = """
    n_peaks: int
        The number of peaks to select
    select_per_channel: bool, default: False
        If True then select n_peaks per channel, else n_peaks in total
    seed: int, default: None
        The random seed for peak selection
    """

    def __init__(self, n_peaks, seed=None, select_per_channel=False):
        self.n_peaks = n_peaks
        self.seed = seed if seed else None
        self.select_per_channel = select_per_channel
        self.rng = np.random.default_rng(seed=self.seed)

    def compute(self, peaks):

        selected_indices = []

        if self.select_per_channel:
            ## This method will randomly select max_peaks_per_channel peaks per channels
            for channel in np.unique(peaks["channel_index"]):
                peaks_indices = np.where(peaks["channel_index"] == channel)[0]
                max_peaks = min(peaks_indices.size, self.n_peaks)
                selected_indices += [self.rng.choice(peaks_indices, size=max_peaks, replace=False)]
        else:
            num_peaks = min(peaks.size, self.n_peaks)
            selected_indices = [self.rng.choice(peaks.size, size=num_peaks, replace=False)]
        
        return selected_indices