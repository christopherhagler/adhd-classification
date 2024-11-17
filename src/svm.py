import numpy as np
from scipy.stats import f_oneway
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def multi_svm_cv_ttest(group1, group2, group3, feature_number):
    # Combine all groups into one dataset
    all_features_X = np.vstack([group1, group2, group3])
    n_subs = all_features_X.shape[0]

    # Assign labels to each group
    labels1 = np.zeros(group1.shape[0], dtype=int)
    labels2 = np.ones(group2.shape[0], dtype=int)
    labels3 = np.full(group3.shape[0], 2, dtype=int)
    all_features_Y = np.concatenate([labels1, labels2, labels3])

    hit_rate = 0
    rank = np.zeros((n_subs, all_features_X.shape[1]), dtype=int)

    for sbj in range(n_subs):
        # Separate the test data
        test_X = all_features_X[sbj, :]
        test_Y = all_features_Y[sbj]

        # Prepare the training data by excluding the test subject
        train_X = np.delete(all_features_X, sbj, axis=0)
        train_Y = np.delete(all_features_Y, sbj)

        # Split training data into groups based on labels
        train_X1 = train_X[train_Y == 0, :]
        train_X2 = train_X[train_Y == 1, :]
        train_X3 = train_X[train_Y == 2, :]

        # Perform ANOVA test for each feature across the three groups
        p_values = np.zeros(train_X.shape[1])
        for i in range(train_X.shape[1]):
            feature_values1 = train_X1[:, i]
            feature_values2 = train_X2[:, i]
            feature_values3 = train_X3[:, i]
            _, p_val = f_oneway(feature_values1, feature_values2, feature_values3)
            p_values[i] = p_val

        # Rank features based on p-values and select top features
        sorted_indices = np.argsort(p_values)
        top_features = sorted_indices[:feature_number]
        rank[sbj, :feature_number] = top_features

        # Prepare training and testing sets with selected features
        trn_X = train_X[:, top_features]
        tst_X = test_X[top_features]

        # Standardize features
        scaler = StandardScaler()
        trn_X_scaled = scaler.fit_transform(trn_X)
        tst_X_scaled = scaler.transform(tst_X.reshape(1, -1))

        # Define and train the SVM model
        svm_model = SVC(kernel='rbf', decision_function_shape='ovr')
        svm_model.fit(trn_X_scaled, train_Y)

        # Make prediction on the test data
        predict_x = svm_model.predict(tst_X_scaled)

        if predict_x[0] == test_Y:
            hit_rate += 1

        print(f"Subject {sbj + 1}/{n_subs}, Feature number: {feature_number}")

    hit_rate = hit_rate / n_subs
    return hit_rate
