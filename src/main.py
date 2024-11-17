import numpy as np
import scipy.io
from scipy.signal import welch
import matplotlib.pyplot as plt

from svm import multi_svm_cv_ttest

HEALTHY = "healthy"
ADHD_TYPE_1 = "adhd_inattentive"
ADHD_TYPE_2 = "adhd_hyperactive"


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


def psd(vector):
    time_series = vector.flatten()
    return welch(time_series, fs=0.5, nperseg=55)


def plot(frequencies, pxx):
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, pxx, color='red', lw=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (PSD)')
    plt.title('Power Spectral Density using Welch\'s Method')
    plt.grid(True)
    plt.show()


def calculate_psd_features():
    group_psd_values = {}

    data: dict[str, np.ndarray] = load_data()
    for classification, group in data.items():
        group_features = []
        for subject in group.flatten():
            psd_region_values = []
            for region in subject.T:
                frequencies, pxx = psd(region)
                psd_region_values.append(pxx)

            group_features.append(np.array(psd_region_values).T)
        group_psd_values[classification] = group_features

    return group_psd_values


def main():
    group_features = calculate_psd_features()

    # Extract features for each class
    features_class0 = group_features[HEALTHY]
    features_class1 = group_features[ADHD_TYPE_1]
    features_class2 = group_features[ADHD_TYPE_2]

    # Ensure all feature arrays have the same number of features
    min_features = min(features_class0.shape[1], features_class1.shape[1], features_class2.shape[1])
    features_class0 = features_class0[:, :min_features]
    features_class1 = features_class1[:, :min_features]
    features_class2 = features_class2[:, :min_features]

    # Define the number of top features to select
    feature_number = 100  # Adjust as needed, must be <= min_features

    # Run the SVM cross-validation
    hit_rate = multi_svm_cv_ttest(features_class0, features_class1, features_class2, feature_number)
    print(f"Hit Rate: {hit_rate * 100:.2f}%")


if __name__ == "__main__":
    main()
