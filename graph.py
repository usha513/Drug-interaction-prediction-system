from os import F_TEST
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pytools import _test

# Load your test dataset here (replace 'your_test_data' with your actual test data)
# X_test and y_test should be your test data and labels.
# Replace 'your_test_data' with your actual test data.

# Load the saved SVM model
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Initialize lists to store accuracy values
accuracies = []

# Vary a parameter or index for different test samples (e.g., sample index)
sample_indices = range(len(y_test))  # Assuming y_test contains labels for test samples

# Calculate accuracy for each test sample
for index in sample_indices:
    # Get a single test sample
    test_sample = F_TEST[index]  # Replace with your actual way of accessing test samples

    # Predict using the model for the current test sample
    y_pred = svm_model.predict([test_sample])  # Note the [test_sample] to create a batch with one sample

    # Compare the predicted label with the actual label (assuming binary classification)
    actual_label = _test[index]  # Replace with your actual way of accessing actual labels
    accuracy = 1 if y_pred[0] == actual_label else 0

    # Append accuracy to the list
    accuracies.append(accuracy)

# Plotting the accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(sample_indices, accuracies, marker='o')
plt.title('Accuracy Graph')
plt.xlabel('Sample Index')
plt.ylabel('Accuracy (1: Correct, 0: Incorrect)')
plt.show()
