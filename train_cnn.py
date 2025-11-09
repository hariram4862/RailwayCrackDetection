
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

X = np.load("fft_data/fft_mags.npy")
y = np.load("fft_data/labels.npy")

X = X[..., np.newaxis] 
y = to_categorical(y, 3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = Sequential([
    Conv1D(16, 5, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(2),
    Conv1D(32, 5, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.3f}")

model.save("models/cnn_fft_model.h5")



# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# import os

# DATA_DIR = "data/fft"
# MODEL_DIR = "models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# X = np.load(os.path.join(DATA_DIR, "fft_data.npy"))
# y = np.load(os.path.join(DATA_DIR, "fft_labels.npy"))

# X = X[..., np.newaxis]  # add channel dimension
# y = to_categorical(y, num_classes=2)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = Sequential([
#     Conv1D(32, 5, activation='relu', input_shape=(256, 1)),
#     MaxPooling1D(2),
#     Dropout(0.2),
#     Conv1D(64, 3, activation='relu'),
#     MaxPooling1D(2),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#     Dense(2, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# history = model.fit(X_train, y_train, epochs=25, batch_size=16, validation_data=(X_test, y_test))

# loss, acc = model.evaluate(X_test, y_test)
# print(f" Test accuracy: {acc*100:.2f}%")

# model.save(os.path.join(MODEL_DIR, "cnn_fft_model.keras"))
# print(" Saved CNN model to models/cnn_fft_model.keras")
