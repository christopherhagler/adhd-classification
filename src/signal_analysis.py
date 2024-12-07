import scipy
from scipy.signal import welch
import numpy as np

HEALTHY = "healthy"
ADHD_TYPE_1 = "adhd_inattentive"
ADHD_TYPE_2 = "adhd_hyperactive"

PEAK_FREQUENCY = 0
AVERAGE_POWER = 1
TOTAL_POWER = 2
SPECTRAL_ENTROPY = 3


def load_data() -> dict[str, np.ndarray]:
    mat_file_path = "src/resources/data.mat"
    data = scipy.io.loadmat(mat_file_path)

    data_class0 = data['data_class0']
    data_class1 = data['data_class1']
    data_class2 = data['data_class2']

    return {
        HEALTHY: data_class0,
        ADHD_TYPE_1: data_class1,
        ADHD_TYPE_2: data_class2
    }


def average_power(time_series, frequency_bins, fs=1.0, nperseg=None):
    # Flatten the time series
    time_series = time_series.flatten()

    # Compute the two-sided Welch PSD
    frequencies, pxx = welch(time_series, fs=fs, nperseg=nperseg, return_onesided=False)

    # Sort frequencies and pxx from negative to positive
    sort_idx = np.argsort(frequencies)
    frequencies = frequencies[sort_idx]
    pxx = pxx[sort_idx]

    avg_powers = []
    for freq_min, freq_max in frequency_bins:
        # Find indices of frequencies within the current bin
        indices = np.where((frequencies >= freq_min) & (frequencies < freq_max))[0]
        if len(indices) == 0:
            # No frequencies in this bin
            avg_powers.append(None)
        else:
            # Calculate the average power in this bin
            pxx_in_bin = pxx[indices]
            avg_power = np.mean(pxx_in_bin)
            avg_powers.append(avg_power)

    return avg_powers


def total_power(time_series, frequency_bins, fs=1.0, nperseg=None):
    # Flatten the time series
    time_series = time_series.flatten()

    # Compute the two-sided Welch PSD
    frequencies, pxx = welch(time_series, fs=fs, nperseg=nperseg, return_onesided=False)

    # Sort frequencies and pxx from negative to positive
    sort_idx = np.argsort(frequencies)
    frequencies = frequencies[sort_idx]
    pxx = pxx[sort_idx]

    total_powers = []
    for freq_min, freq_max in frequency_bins:
        # Find indices of frequencies within the current bin
        indices = np.where((frequencies >= freq_min) & (frequencies < freq_max))[0]

        if len(indices) == 0:
            # No frequencies in this bin
            total_powers.append(None)
        else:
            # Calculate total power in the bin using trapezoidal integration
            freqs_in_bin = frequencies[indices]
            pxx_in_bin = pxx[indices]
            total_power_bin = np.trapz(pxx_in_bin, freqs_in_bin)
            total_powers.append(total_power_bin)

    return total_powers


def peak_frequency(time_series, frequency_bins, fs=1.0, nperseg=None):
    # Flatten the time series
    time_series = time_series.flatten()

    # Compute the two-sided Welch PSD
    # Note: return_onesided=False returns negative frequencies as well
    frequencies, pxx = welch(time_series, fs=fs, nperseg=nperseg, return_onesided=False)

    # The returned frequencies for a two-sided PSD from Welch are typically arranged as:
    # [0, 1*df, 2*df, ... , (N/2)*df, -N/2*df, ... , -2*df, -1*df]
    # We want to sort them from negative to positive for intuitive indexing.
    sort_idx = np.argsort(frequencies)
    frequencies = frequencies[sort_idx]
    pxx = pxx[sort_idx]

    peak_freqs = []
    for freq_min, freq_max in frequency_bins:
        # Find indices of frequencies within the current bin
        indices = np.where((frequencies >= freq_min) & (frequencies < freq_max))[0]

        if len(indices) == 0:
            # No frequencies in this bin
            peak_freqs.append(None)
        else:
            # Find the frequency with the maximum power in this bin
            pxx_in_bin = pxx[indices]
            max_idx = np.argmax(pxx_in_bin)
            peak_freq = frequencies[indices[max_idx]]
            peak_freqs.append(peak_freq)

    return peak_freqs


def spectral_entropy(time_series, frequency_bins=None):
    """
    Compute spectral entropy over specified frequency bins, including negative frequencies.
    Returns a vector of entropy values, one for each bin.
    """
    time_series = time_series.flatten()

    # Compute Welch's PSD with two-sided spectrum
    frequencies, pxx = welch(time_series, return_onesided=False)

    # Shift frequencies and PSD for two-sided spectrum
    frequencies = np.fft.fftshift(frequencies)
    pxx = np.fft.fftshift(pxx)

    # Compute total power
    total_power = np.sum(pxx)
    if total_power == 0:
        return np.zeros(len(frequency_bins))

    entropy_values = []
    for f_min, f_max in frequency_bins:
        # Select frequencies within the current bin
        bin_mask = (frequencies >= f_min) & (frequencies < f_max)
        pxx_bin = pxx[bin_mask]
        bin_power = np.sum(pxx_bin)

        if bin_power == 0:
            # No power in this bin; entropy is zero
            entropy_values.append(0)
            continue

        # Normalize power within the bin
        pxx_normalized = pxx_bin / bin_power

        # Compute entropy for the current bin
        entropy_bin = -np.sum(pxx_normalized * np.log2(pxx_normalized + 1e-12))
        entropy_values.append(entropy_bin)

    return np.array(entropy_values)


def calculate_features(selection: int = PEAK_FREQUENCY):
    features = {}
    frequency_bins = [
        (-0.50, -0.45),
        (-0.45, -0.40),
        (-0.40, -0.35),
        (-0.35, -0.30),
        (-0.30, -0.25),
        (-0.25, -0.20),
        (-0.20, -0.15),
        (-0.15, -0.10),
        (-0.10, -0.05),
        (-0.05, 0.00),
        (0.00, 0.05),
        (0.05, 0.10),
        (0.10, 0.15),
        (0.15, 0.20),
        (0.20, 0.25),
        (0.25, 0.30),
        (0.30, 0.35),
        (0.35, 0.40),
        (0.40, 0.45),
        (0.45, 0.50),
    ]

    data: dict[str, np.ndarray] = load_data()
    for classification, group in data.items():
        group_psd_values = []
        for subject in group.flatten():
            region_values = []
            for region in subject.T:
                if selection == PEAK_FREQUENCY:
                    region_values.extend(peak_frequency(time_series=region, frequency_bins=frequency_bins))
                elif selection == AVERAGE_POWER:
                    region_values.extend(average_power(time_series=region, frequency_bins=frequency_bins))
                elif selection == SPECTRAL_ENTROPY:
                    region_values.extend(spectral_entropy(time_series=region, frequency_bins=frequency_bins))
                else:
                    region_values.extend(total_power(time_series=region, frequency_bins=frequency_bins))

            group_psd_values.append(region_values)
        features[classification] = np.array(group_psd_values)

    return features
