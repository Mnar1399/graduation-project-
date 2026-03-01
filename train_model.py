import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# تحميل البيانات
X = np.load("X_multi.npy")
y = np.load("y_multi.npy")

# عدد الجمل
num_classes = len(np.unique(y))

# تحويل labels إلى one-hot
y = to_categorical(y, num_classes=num_classes)

print("X shape:", X.shape)
print("y shape:", y.shape)

# بناء المودل الأساسي
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(30, 63)))
model.add(LSTM(64))
model.add(Dense(64, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# تدريب
model.fit(X, y, epochs=30, batch_size=8)

# حفظ المودل
model.save("sign_model.h5")

print("Model training finished!")