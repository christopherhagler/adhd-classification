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


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def average_power(time_series):
    time_series = time_series.flatten()
    _, pxx = welch(time_series, fs=0.5, nperseg=55)
    return np.mean(pxx)


def peak_frequency(time_series):
    time_series = time_series.flatten()
    frequencies, pxx = welch(time_series, fs=0.5, nperseg=55)
    return frequencies[np.argmax(pxx)]


def band_power(time_series, f_min, f_max):
    frequencies, pxx = welch(time_series, fs=0.5, nperseg=55)
    band = (frequencies >= f_min) & (frequencies <= f_max)
    return np.sum(pxx[band])


def spectral_entropy(time_series):
    time_series = time_series.flatten()
    frequencies, pxx = welch(time_series, fs=0.5, nperseg=55)
    total_power = np.sum(pxx)
    if total_power == 0:
        return 0

    pxx_normalized = pxx / total_power
    spectral_entropy_value = -np.sum(pxx_normalized * np.log2(pxx_normalized + 1e-12))
    return spectral_entropy_value


def calculate_features(selection: int = PEAK_FREQUENCY):
    features = {}

    data: dict[str, np.ndarray] = load_data()
    for classification, group in data.items():
        group_psd_values = []
        for subject in group.flatten():
            region_values = []
            for region in subject.T:
                if selection == PEAK_FREQUENCY:
                    region_values.append(peak_frequency(region))
                elif selection == AVERAGE_POWER:
                    region_values.append(average_power(region))
                else:
                    region_values.append(spectral_entropy(region))

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
    feature_numbers = range(1, min_features + 1, 9)
    hit_rates = []

    # for feature_number in feature_numbers:
    hit_rate = multi_svm_cv_ttest(features_class0, features_class1, features_class2, 100)
    hit_rates.append(hit_rate * 100)  # Store hit rate as a percentage
    print(f"Feature number: {100}, Hit Rate: {hit_rate * 100:.2f}%")

    # plt.figure(figsize=(10, 6))
    # plt.plot(feature_numbers, hit_rates, marker='o', linestyle='-', color='b')
    # plt.xlabel('Number of Top Features')
    # plt.ylabel('Hit Rate (%)')
    # plt.title('Hit Rate vs. Number of Top Features (Spectral Entropy)')
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    main()
