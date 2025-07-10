
# 🧠 MNIST Digit Classification using TensorFlow/Keras

This project demonstrates a simple **Feed-Forward Neural Network (FFNN)** built with TensorFlow/Keras for classifying handwritten digits from the **MNIST dataset**.

---

## 📦 Requirements

Make sure you have the following Python libraries installed:

```bash
pip install tensorflow numpy matplotlib
```

If you're using a conda environment, create and activate it first:

```bash
conda create -n myenv python=3.10
conda activate myenv
```

---

## 📊 Dataset

We use the MNIST dataset:
- 60,000 training images
- 10,000 test images
- Each image is 28x28 pixels (grayscale)

---

## 📁 Project Structure

```
mnist-digit-classification/
│
├── mnist_model.ipynb         # Jupyter Notebook with full code
├── README.md                 # This file
```

---

## 🏗️ Model Architecture

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

---

## 🏋️‍♂️ Training

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    validation_split=0.2)
```

---

## 📈 Evaluation & Visualization

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)
```

```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
```

---

## 🔍 Prediction

```python
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)
actual_labels = np.argmax(y_test[:5], axis=1)

for i in range(5):
    image = x_test[i].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted_labels[i]}, Actual: {actual_labels[i]}")
    plt.axis('off')
    plt.show()
```

---

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MNIST Dataset Info](http://yann.lecun.com/exdb/mnist/)

---

🎉 Happy Learning!
