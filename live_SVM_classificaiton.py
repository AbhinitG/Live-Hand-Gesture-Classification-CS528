import os
import serial
import re
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Function to remove ANSI escape codes from strings
def remove_ansi_escape_codes(line):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', line)

# Function to extract relevant data from the log line
def extract_relevant_data(line):
    # Match lines containing "acce" or "gyro" after log prefixes
    match = re.search(r'(acce,.*|gyro,.*)', line)
    return match.group(1) if match else None

# Function to preprocess data for live classification
def preprocess_data(acce_data, gyro_data):
    # Combine accelerometer and gyroscope data
    acce_values = np.array([list(map(float, row.split(',')[1:])) for row in acce_data])
    gyro_values = np.array([list(map(float, row.split(',')[1:])) for row in gyro_data])

    combined_data = np.hstack((acce_values, gyro_values))

    # Normalize data
    combined_data = (combined_data - combined_data.min(axis=0)) / (combined_data.max(axis=0) - combined_data.min(axis=0))

    return combined_data.flatten().reshape(1, -1)

# Function to classify gesture
def classify_gesture(svm_classifier, data):
    prediction = svm_classifier.predict(data)
    return prediction[0]

# Function to load dataset
def load_data(label_df):
    features = []
    labels = []

    for _, row in label_df.iterrows():
        try:
            # Read the combined CSV file
            df = pd.read_csv(row["filename"])

            columns = ["acce_x", "acce_y", "acce_z", "gyro_x", "gyro_y", "gyro_z"]
            if not all(col in df.columns for col in columns):
                raise ValueError(f"Missing required columns in file: {row['filename']}")

            # Extract accelerometer and gyroscope signals
            data = df[columns].values.astype(np.float32)

            if np.isnan(data).sum() > 0:
                raise ValueError(f"File contains NaN values before normalization: {row['filename']}")

            # Normalize data
            data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

            # Flatten the data and add to the list
            features.append(data.flatten())
            labels.append(row["label"])

        except Exception as e:
            print(f"Error processing file: {row['filename']}")
            print(f"Error details: {e}")
            continue  # Skip the problematic file and move to the next one

    return np.array(features), np.array(labels)

def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    # Create the SVM classifier
    svm_classifier = SVC(kernel="rbf", probability=True)

    # Train the classifier
    svm_classifier.fit(X_train, y_train)

    # Saving the trained model
    joblib.dump(svm_classifier, "svm_model.joblib")

    # Perform prediction on the test set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM accuracy: {accuracy:.3%}")

    # Plot the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=["up", "down", "left", "right", "none"])
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=["up", "down", "left", "right", "none"], yticklabels=["up", "down", "left", "right", "none"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

train_labels = pd.read_csv("train_labels.csv") # Update directory
val_labels = pd.read_csv("val_labels.csv") # Update directory
# Create the train and test sets
X_train, y_train = load_data(train_labels)
X_test, y_test = load_data(val_labels)
# Perform training and testing with SVM
train_and_evaluate_svm(X_train, y_train, X_test, y_test)

def live_classification():
    # Replace with the correct serial port and baud rate
    SERIAL_PORT = 'COM4'  # Adjust to your port
    BAUD_RATE = 115200
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    acce_data = []
    gyro_data = []

    try:
        print("Starting live gesture classification...")
        while True:
            if ser.in_waiting:
                # Read a line from the serial port
                line = ser.readline().decode('utf-8').strip()

                # Remove ANSI escape codes
                line = remove_ansi_escape_codes(line)

                # Extract accelerometer or gyroscope data
                relevant_data = extract_relevant_data(line)
                if not relevant_data:
                    continue  # Skip lines that don't contain relevant data

                # Parse and save accelerometer and gyroscope data
                if relevant_data.startswith("acce"):
                    acce_data.append(relevant_data)
                elif relevant_data.startswith("gyro"):
                    gyro_data.append(relevant_data)

                if len(acce_data) == len(gyro_data):
                    print(f'Count: {len(acce_data)}')

                # Predict the gesture after having 400 data points of both sensors
                if len(acce_data) >= 400 and len(gyro_data) >= 400:
                    # Preprocess data
                    data = preprocess_data(acce_data[:400], gyro_data[:400])

                    # Predict the gesture
                    gesture = classify_gesture(svm_classifier=joblib.load("svm_model.joblib"), data=data)
                    print(f"Predicted Gesture: {gesture}")

                    # Clear the data buffers for next classification
                    acce_data = []
                    gyro_data = []

    except KeyboardInterrupt:
        print("Ending live classification...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ser.close()

# Starting live classification
if __name__ == "__main__":
    live_classification()