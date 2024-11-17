# Multi-Class SVM Cross-Validation with Feature Selection via Welch's PSD Method

This project implements a multi-class Support Vector Machine (SVM) classifier with cross-validation, using features extracted via Welch's power spectral density (PSD) method. The goal is to classify data from different brain regions using features derived from their fMRI time series.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Configuration](#configuration)
- [Example Usage](#example-usage)
- [Contact Information](#contact-information)

## Project Overview

- **Objective**: To classify fMRI data from different brain regions using an SVM model with features extracted through Welch's PSD method.
- **Methodology**:
  - Extract power spectral density features from fMRI time series using Welch's method.
  - Standardize features for better performance.
  - Train an SVM model using Leave-One-Out Cross-Validation (LOOCV).
  - Assess model accuracy.

## Directory Structure

```
project-root/
├── pyproject.toml
├── README.md
├── src/
│   ├── main.py
│   └── svm.py
└── test/
    └── test_svm.py
```

- `pyproject.toml`: Configuration file for project dependencies and scripts.
- `README.md`: This readme file.
- `src/`: Contains the main code.
  - `main.py`: Entry point for running the project.
  - `svm.py`: Contains the `multi_svm_cv_ttest` function for SVM cross-validation.
- `test/`: Contains test scripts for validating the SVM implementation.

## Requirements

- **Python**: 3.8 or higher
- **Packages**:
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `pytest` (for testing)

All dependencies are specified in the `pyproject.toml` file and can be installed using [Poetry](https://python-poetry.org/).

## Installation

1. **Clone the Repository**

   Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/adhd-classification.git
   cd adhd-classification
   ```

2. **Install Poetry**

   If you don't have Poetry installed, you can install it using:

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   For alternative installation methods, refer to the [Poetry Documentation](https://python-poetry.org/docs/#installation).

3. **Install Project Dependencies**

   Install the required dependencies using Poetry:

   ```bash
   poetry install
   ```

   This command creates a virtual environment and installs all dependencies specified in `pyproject.toml`.

## Running the Project

1. **Activate the Virtual Environment**

   Before running the project, activate the virtual environment:

   ```bash
   poetry shell
   ```

2. **Run the Main Script**

   Run the project using the following command:

   ```bash
   poetry run classification
   ```

   This will execute the SVM cross-validation process and output the classification accuracy.

## Testing

To ensure that the code works correctly, you can run the test script provided.

1. **Run the Test Script**

   From the project root directory, execute:

   ```bash
   pytest test/
   ```

   This will run the test script in the `test/` directory.

## Configuration

- **Feature Selection**: The number of top features to select is controlled by the `feature_number` variable in `main.py`. Adjust this number based on your needs.
- **SVM Parameters**: The SVM model uses an RBF kernel. You can modify parameters in `svm.py` if needed.

## Example Usage

Below is an example of how to use the `multi_svm_cv_ttest` function in your `main.py` file.

```python
# src/main.py
import numpy as np
from svm import multi_svm_cv_ttest

def main():
    # Example data
    group1 = np.random.rand(10, 100)
    group2 = np.random.rand(10, 100)
    group3 = np.random.rand(10, 100)

    feature_number = 50
    hit_rate = multi_svm_cv_ttest(group1, group2, group3, feature_number)
    print(f"Hit Rate: {hit_rate * 100:.2f}%")

if __name__ == "__main__":
    main()
```

Make sure to replace the synthetic data with your actual datasets.

## Contact Information

For any questions or issues, please contact:

- **Name**: Christopher Hagler
- **Email**: cwh0020@auburn.edu

---

Thank you for using this project. Good luck!


