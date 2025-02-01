# trainer/task.py

import argparse
import os
import tensorflow as tf

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    
    # Common Vertex arguments:
    # --model_dir, --epochs, --batch_size, etc.
    parser.add_argument(
        "--model_dir",
        type=str,
        default="gs://YOUR_BUCKET/YOUR_OUTPUT_PATH",
        help="GCS or local directory to store model artifacts"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--flag_a",
        type=str,
        default=None,
        help="Example custom argument from the console: e.g., --flag_a=xxxx"
    )
    # Add more arguments as needed...

    return parser.parse_args()

def main():
    # 1. Parse arguments
    args = get_args()
    print("Training arguments:", args)

    # 2. Load or generate your dataset
    # For example, let's do a simple Keras model on MNIST:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape([-1, 28, 28, 1]) / 255.0
    x_test = x_test.reshape([-1, 28, 28, 1]) / 255.0

    # 3. Build a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Train the model
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # 5. Evaluate
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Final test accuracy: {acc}")

    # 6. Save model artifacts to GCS or local path
    # Note that Vertex sets a default --model_dir if you do not override
    model.save(os.path.join(args.model_dir, "saved_model"))
    print(f"Model saved to {args.model_dir}/saved_model")

    # 7. (Optional) Handle any custom logic here, e.g. writing metrics, etc.
    if args.flag_a:
        print(f"Custom flag was provided: {args.flag_a}")

if __name__ == "__main__":
    main()
