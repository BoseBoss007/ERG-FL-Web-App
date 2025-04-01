from flask import Flask, request, jsonify
import json
import numpy as np
import tensorflow as tf
import threading

app = Flask(__name__)

# Global model
global_model = None
client_weights = []
client_count = 0
lock = threading.Lock()  # Thread-safe updates

@app.route("/update", methods=["POST"])
def update_model():
    global global_model, client_weights, client_count

    try:
        data = request.get_json()

        if not data or "weights" not in data:
            return jsonify({"error": "Invalid request! Missing weights."}), 400

        received_weights = [np.array(w) for w in json.loads(data["weights"])]

        with lock:
            client_weights.append(received_weights)
            client_count += 1

        # Wait until all clients have sent their updates before averaging
        if client_count >= 2:  # Change this number to match the expected number of clients
            print("Aggregating weights using FedAvg...")

            # Initialize averaged weights with zeros
            avg_weights = [np.zeros_like(w) for w in client_weights[0]]

            # Compute FedAvg: Average the weights from all clients
            for w_set in client_weights:
                for i in range(len(avg_weights)):
                    avg_weights[i] += w_set[i] / client_count  # Element-wise averaging

            # Set the averaged weights as the new global model weights
            global_model.set_weights(avg_weights)

            # Clear client weights after aggregation
            client_weights.clear()
            client_count = 0

            print("Global model updated with FedAvg.")

        # Return the (potentially updated) global model weights
        response_data = json.dumps([w.tolist() for w in global_model.get_weights()])
        return jsonify({"global_weights": response_data})

    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({"error": str(e)}), 500

def build_model(input_size=8, num_classes=6):
    """
    Builds a simple Neural Network model for classification.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    global_model = build_model(input_size=8, num_classes=5)  # Includes "Healthy"
    print("Global model initialized.")
    app.run(host="0.0.0.0", port=5000, debug=True)
