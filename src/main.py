import math
import os
from pathlib import Path

from matplotlib import pyplot as plt
from signal_analysis import (
    HEALTHY,
    ADHD_TYPE_1,
    ADHD_TYPE_2,
    calculate_features,
    SPECTRAL_ENTROPY,
    PEAK_FREQUENCY,
    AVERAGE_POWER,
    TOTAL_POWER
)
from svm import multi_svm_cv_ttest


def main():
    current_file = Path(__file__).resolve()
    resources_dir = current_file.parent / 'resources' / 'plots'

    plot_hit_rate(SPECTRAL_ENTROPY, resources_dir)
    plot_hit_rate(PEAK_FREQUENCY, resources_dir)
    plot_hit_rate(AVERAGE_POWER, resources_dir)
    plot_hit_rate(TOTAL_POWER, resources_dir)


def plot_hit_rate(selection, output_folder):
    group_features_1 = calculate_features(selection)
    # Extract features for each class
    features_class0 = group_features_1[HEALTHY]
    features_class1 = group_features_1[ADHD_TYPE_1]
    features_class2 = group_features_1[ADHD_TYPE_2]

    min_features = min(features_class0.shape[1], features_class1.shape[1], features_class2.shape[1])
    features_class0 = features_class0[:, :min_features]
    features_class1 = features_class1[:, :min_features]
    features_class2 = features_class2[:, :min_features]

    step = math.floor((min_features + 1) / 5) if (min_features + 1) > 5 else 1
    feature_numbers = range(1, min_features + 1, step)

    hit_rates = []
    for feature_number in feature_numbers:
        hit_rate = multi_svm_cv_ttest(features_class0, features_class1, features_class2, feature_number)
        hit_rates.append(hit_rate * 100)
        print(f"Feature number: {feature_number}, Hit Rate: {hit_rate * 100:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(feature_numbers, hit_rates, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Top Features')
    plt.ylabel('Hit Rate (%)')
    plt.title('Hit Rate vs. Number of Top Features')
    plt.grid(True)

    output_path = os.path.join(output_folder, 'hit_rate_vs_features.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
