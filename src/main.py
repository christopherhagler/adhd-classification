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


def mean_psd(vector):
    time_series = vector.flatten()
    _, pxx = welch(time_series, fs=0.5, nperseg=55)
    return np.mean(pxx)


def calculate_psd_features():
    features = {}

    data: dict[str, np.ndarray] = load_data()
    for classification, group in data.items():
        group_psd_values = []
        for subject in group.flatten():
            average_region_values = []
            for region in subject.T:
                average_region_values.append(mean_psd(region))

            group_psd_values.append(average_region_values)
        features[classification] = np.array(group_psd_values)

    return features


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

    # Define the range of feature numbers to test
    feature_numbers = range(1, min_features + 1, 10)
    hit_rates = []

    # Run SVM cross-validation for each feature number and record hit rate
    for feature_number in feature_numbers:
        hit_rate = multi_svm_cv_ttest(features_class0, features_class1, features_class2, feature_number)
        hit_rates.append(hit_rate * 100)  # Store hit rate as a percentage
        print(f"Feature number: {feature_number}, Hit Rate: {hit_rate * 100:.2f}%")

    # Plot hit rate vs feature number
    plt.figure(figsize=(10, 6))
    plt.plot(feature_numbers, hit_rates, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Top Features')
    plt.ylabel('Hit Rate (%)')
    plt.title('Hit Rate vs. Number of Top Features')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
