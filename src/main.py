import math
from matplotlib import pyplot as plt

from signal import HEALTHY, ADHD_TYPE_1, ADHD_TYPE_2, calculate_features, SPECTRAL_ENTROPY
from svm import multi_svm_cv_ttest


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
    feature_numbers = range(1, min_features + 1, math.floor((min_features + 1) / 5))
    hit_rates = []

    for feature_number in feature_numbers:
        hit_rate = multi_svm_cv_ttest(features_class0, features_class1, features_class2, feature_number)
        hit_rates.append(hit_rate * 100)  # Store hit rate as a percentage
        print(f"Feature number: {feature_number}, Hit Rate: {hit_rate * 100:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(feature_numbers, hit_rates, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Top Features')
    plt.ylabel('Hit Rate (%)')
    plt.title('Hit Rate vs. Number of Top Features')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
