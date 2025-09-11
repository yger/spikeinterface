import numpy as np
import

class SmartSamplingByAmplitudes:
    
    name = "smart_sampling_amplitudes"
    need_noise_levels = True
    params_doc = """
    n_peaks: int
        The number of peaks to select
    noise_levels: array
        The noise levels for each channel
    select_per_channel: bool, default: False
        If True then select n_peaks per channel, else n_peaks in total
    seed: int, default: None
        The random seed for peak selection
    """

    def __init__(self, n_peaks, noise_levels, seed=None, select_per_channel=False):
        self.n_peaks = n_peaks
        self.noise_levels = noise_levels
        self.seed = seed if seed else None
        self.select_per_channel = select_per_channel
        self.rng = np.random.default_rng(seed=self.seed)

    def compute(self, peaks):
        
        from sklearn.preprocessing import QuantileTransformer
        selected_indices = []

        if self.select_per_channel:
            for channel in np.unique(peaks["channel_index"]):
                peaks_indices = np.where(peaks["channel_index"] == channel)[0]
                if self.n_peaks > peaks_indices.size:
                    selected_indices += [peaks_indices]
                else:
                    sub_peaks = peaks[peaks_indices]
                    snrs = sub_peaks["amplitude"] / self.noise_levels[channel]
                    preprocessing = QuantileTransformer(
                        output_distribution="uniform", n_quantiles=min(100, len(snrs))
                    )
                    snrs = preprocessing.fit_transform(snrs[:, np.newaxis])

                    my_selection = np.zeros(0, dtype=np.int32)
                    all_index = np.arange(len(snrs))
                    while my_selection.size < self.n_peaks:
                        candidates = all_index[np.logical_not(np.isin(all_index, my_selection))]
                        probabilities = self.rng.random(size=len(candidates))
                        valid = candidates[np.where(snrs[candidates, 0] < probabilities)[0]]
                        my_selection = np.concatenate((my_selection, valid))

                    selected_indices += [peaks_indices[self.rng.permutation(my_selection)[: self.n_peaks]]]
        else:
            if self.n_peaks > peaks.size:
                selected_indices += [np.arange(peaks.size)]
            else:
                snrs = peaks["amplitude"] / self.noise_levels[peaks["channel_index"]]
                preprocessing = QuantileTransformer(output_distribution="uniform", n_quantiles=min(100, len(snrs)))
                snrs = preprocessing.fit_transform(snrs[:, np.newaxis])

                my_selection = np.zeros(0, dtype=np.int32)
                all_index = np.arange(len(snrs))
                while my_selection.size < self.n_peaks:
                    candidates = all_index[np.logical_not(np.isin(all_index, my_selection))]
                    probabilities = self.rng.random(size=len(candidates))
                    valid = candidates[np.where(snrs[candidates, 0] < probabilities)[0]]
                    my_selection = np.concatenate((my_selection, valid))

                selected_indices = [self.rng.permutation(my_selection)[: self.n_peaks]]

        return selected_indices


class SmartSamplingByLocations:

    name = "smart_sampling_locations"
    need_noise_levels = True
    params_doc = """
    n_peaks: int
        The number of peaks to select
    noise_levels: array
        The noise levels for each channel
    select_per_channel: bool, default: False
        If True then select n_peaks per channel, else n_peaks in total
    seed: int, default: None
        The random seed for peak selection
    """

    def __init__(self, n_peaks, noise_levels, peak_locations, seed=None, select_per_channel=False):
        self.n_peaks = n_peaks
        self.noise_levels = noise_levels
        self.peak_locations = peak_locations
        self.seed = seed if seed else None
        self.select_per_channel = select_per_channel
        self.rng = np.random.default_rng(seed=self.seed)

    def compute(self, peaks):

        from sklearn.preprocessing import QuantileTransformer
        selected_indices = []

        nb_spikes = len(peaks)

        if self.n_peaks > nb_spikes:
            selected_indices += [np.arange(peaks.size)]
        else:
            preprocessing = QuantileTransformer(output_distribution="uniform", n_quantiles=min(100, nb_spikes))
            data = np.array([self.peak_locations["x"], self.peak_locations["y"]]).T
            data = preprocessing.fit_transform(data)

            my_selection = np.zeros(0, dtype=np.int32)
            all_index = np.arange(peaks.size)
            while my_selection.size < self.n_peaks:
                candidates = all_index[np.logical_not(np.isin(all_index, my_selection))]

                probabilities = self.rng.random(size=len(candidates))
                data_x = data[:, 0] < probabilities

                probabilities = self.rng.random(size=len(candidates))
                data_y = data[:, 1] < probabilities

                valid = candidates[np.where(data_x * data_y)[0]]
                my_selection = np.concatenate((my_selection, valid))

            selected_indices = [self.rng.permutation(my_selection)[: self.n_peaks]]
        return selected_indices