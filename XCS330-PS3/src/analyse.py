import numpy as np
import matplotlib.pyplot as plt

results_protonet_1_5 = np.load('/Users/pat-home/Desktop/Stanford Generative AI/Assignment 1/XCS330-PS3/src/submission/protonet_results_1_5.npy', allow_pickle=True)
results_protonet_5_5 = np.load('/Users/pat-home/Desktop/Stanford Generative AI/Assignment 1/XCS330-PS3/src/submission/protonet_results_5_5.npy', allow_pickle=True)

print("Results for 1-shot 5-way:")
print(results_protonet_1_5)

print("\nResults for 5-shot 5-way:")
print(results_protonet_5_5)

def plot_metrics(metrics, title):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_accuracy'], label='Train Accuracy')
    plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
    plt.plot(metrics['test_accuracy'], label='Test Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if isinstance(results_protonet_1_5, float):
    print(f"1-shot 5-way metric: {results_protonet_1_5}")

if isinstance(results_protonet_5_5, float):
    print(f"5-shot 5-way metric: {results_protonet_5_5}")


import tensorflow as tf

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])

