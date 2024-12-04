import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from scipy.signal import welch, butter, filtfilt
from svm import multi_svm_cv_ttest

HEALTHY = "healthy"
ADHD_TYPE_1 = "adhd_inattentive"
ADHD_TYPE_2 = "adhd_hyperactive"

PEAK_FREQUENCY = 0
AVERAGE_POWER = 1
BAND_POWER = 2
SPECTRAL_ENTROPY = 3
SPECTRAL_DENSITY = 4


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


def average_power(time_series):
    time_series = time_series.flatten()
    _, pxx = welch(time_series, fs=0.5, nperseg=55)
    return np.mean(pxx)


def total_power(time_series):
    time_series = time_series.flatten()
    _, pxx = welch(time_series)
    return pxx


def peak_frequency(time_series):
    time_series = time_series.flatten()
    frequencies, pxx = welch(time_series)
    return frequencies[np.argmax(pxx)]


def band_power(time_series, f_min, f_max):
    frequencies, pxx = welch(time_series)
    band = (frequencies >= f_min) & (frequencies <= f_max)
    return np.sum(pxx[band])


def spectral_entropy(time_series, frequency_bins=None):
    """
        Compute spectral entropy over specified frequency bins.
        Returns a vector of entropy values, one for each bin.
        """
    time_series = time_series.flatten()
    frequencies, pxx = welch(time_series)
    total_power = np.sum(pxx)
    if total_power == 0:
        return np.zeros(len(frequency_bins))

    entropy_values = []
    for f_min, f_max in frequency_bins:
        bin_mask = (frequencies >= f_min) & (frequencies < f_max)
        pxx_bin = pxx[bin_mask]
        bin_power = np.sum(pxx_bin)
        if bin_power == 0:
            entropy_values.append(0)
            continue

        pxx_normalized = pxx_bin / bin_power
        entropy_bin = -np.sum(pxx_normalized * np.log2(pxx_normalized + 1e-12))
        entropy_values.append(entropy_bin)
    return np.array(entropy_values)


def calculate_features(selection: int = PEAK_FREQUENCY):
    features = {}

    data: dict[str, np.ndarray] = load_data()
    for classification, group in data.items():
        group_psd_values = []
        for subject in group.flatten():
            region_values = []
            for region in subject.T:
                if selection == PEAK_FREQUENCY:
                    region_values.append(peak_frequency(time_series=region))
                elif selection == AVERAGE_POWER:
                    region_values.append(average_power(time_series=region))
                elif selection == SPECTRAL_ENTROPY:
                    region_values.extend(
                        spectral_entropy(
                            time_series=region,
                            frequency_bins=[
                                (0.00, 0.05),
                                (0.05, 0.10),
                                (0.10, 0.15),
                                (0.15, 0.20),
                                (0.20, 0.25)
                            ]
                        )
                    )
                else:
                    region_values.extend(total_power(time_series=region))

            group_psd_values.append(region_values)
        features[classification] = np.array(group_psd_values)

    return features


def main():
    group_features_1 = calculate_features(SPECTRAL_ENTROPY)

    # Extract features for each class
    features_class0 = group_features_1[HEALTHY]
    features_class1 = group_features_1[ADHD_TYPE_1]
    features_class2 = group_features_1[ADHD_TYPE_2]

    # Ensure all feature arrays have the same number of features
    min_features = min(features_class0.shape[1], features_class1.shape[1], features_class2.shape[1])
    features_class0 = features_class0[:, :min_features]
    features_class1 = features_class1[:, :min_features]
    features_class2 = features_class2[:, :min_features]

    # Define the range of feature numbers to test
    feature_numbers = range(1, min_features + 1, 100)
    hit_rates = []

    hit_rate = multi_svm_cv_ttest(features_class0, features_class1, features_class2, feature_numbers)
    hit_rates.append(hit_rate * 100)  # Store hit rate as a percentage
    print(f"Feature number: {feature_numbers}, Hit Rate: {hit_rate * 100:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(feature_numbers, hit_rates, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Top Features')
    plt.ylabel('Hit Rate (%)')
    plt.title('Hit Rate vs. Number of Top Features (Total Power)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
