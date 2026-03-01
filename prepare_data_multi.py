import numpy as np
import os

DATA_PATH = "data/landmarks"
SEQUENCE_LENGTH = 30

X = []
y = []

labels = sorted(os.listdir(DATA_PATH))
label_map = {label: num for num, label in enumerate(labels)}

print("Label mapping:", label_map)

for label in labels:
    label_folder = os.path.join(DATA_PATH, label)

    for file in os.listdir(label_folder):
        seq = np.load(os.path.join(label_folder, file))

        if len(seq) >= SEQUENCE_LENGTH:
            seq = seq[:SEQUENCE_LENGTH]
        else:
            padding = np.zeros((SEQUENCE_LENGTH - len(seq), 63))
            seq = np.vstack((seq, padding))

        X.append(seq)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

np.save("X_multi.npy", X)
np.save("y_multi.npy", y)