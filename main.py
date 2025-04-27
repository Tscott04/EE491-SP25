import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# STEP 1: Generate synthetic waveform dataset
def generate_wave(class_label):
    t = np.linspace(0, 1, 500)
    if class_label == 0:
        wave = np.sin(2 * np.pi * 5 * t)  # Clean sine wave
    else:
        wave = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(500)  # Noisy sine wave
    return wave

X = []
y = []
for i in range(1000):
    label = np.random.randint(0, 2)
    X.append(generate_wave(label))
    y.append(label)

X = np.array(X)
y = np.array(y)

# STEP 2: Normalize the waveform values
scaler = StandardScaler()
X = np.array([scaler.fit_transform(sample.reshape(-1, 1)).flatten() for sample in X])

# STEP 3: Prepare dataset for TensorFlow (reshape for Conv1D)
X = X[..., np.newaxis]  # Add channel dimension: (samples, 500, 1)
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# STEP 4: Define a simple 1D CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(16, 5, activation='relu', input_shape=(500, 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# STEP 5: Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# STEP 6: Plot training history
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# STEP 7: Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# OPTIONAL: Visualize example waveform
plt.plot(np.linspace(0, 1, 500), X[0].flatten())
plt.title(f"Example Normalized Waveform (Label: {y[0]})")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Amplitude")
plt.grid(True)
plt.show()
