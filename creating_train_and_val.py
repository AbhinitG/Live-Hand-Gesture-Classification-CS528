import os
import pandas as pd
from sklearn.model_selection import train_test_split

# the directories for each gesture
gesture_dirs = ["up", "down", "left", "right", "none"]

# Initialize lists to store filenames and labels
data = []

# Loop through each gesture directory
for gesture in gesture_dirs:
    gesture_path = os.path.join(gesture, "combined")  # Adjust for your directory structure
    files = [f for f in os.listdir(gesture_path) if f.endswith(".csv")]

    # Add each file and its label
    for file in files:
        data.append({"filename": os.path.join(gesture_path, file), "label": gesture})

# Convert the data into a DataFrame
data_df = pd.DataFrame(data)

# Split into train and validation sets
train_df, val_df = train_test_split(data_df, test_size=0.2, stratify=data_df["label"], random_state=42)

# Save the labels for train and validation sets
train_df.to_csv("train_labels.csv", index=False)
val_df.to_csv("val_labels.csv", index=False)

print("Train and validation labels created!")
