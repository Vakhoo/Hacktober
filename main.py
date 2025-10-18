import base64
from flask import Flask, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import pprint
import sys

# --- 1. Load the model ONCE when the app starts ---
# This is much more efficient.
model = YOLO("yolov8m-oiv7.pt")

# Create the Flask application instance
app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins (for hackathon/testing)
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# In-memory storage (this is fine, but unused by the POST route)
greetings = {
    1: "Hello, World!",
    2: "¡Hola, Mundo!"
}

# GET endpoint: retrieves all greetings
@app.route('/greetings', methods=['GET'])
def get_all_greetings():
    return jsonify(list(greetings.values()))


# --- 2. Renamed route to be more descriptive ---
@app.route('/detect', methods=['POST'])
def detect_objects():
    frame_data = request.json.get('frame')

    if not frame_data:
        return jsonify({"error": "No 'frame' key found in JSON payload"}), 400

    try:
        img_data = base64.b64decode(request.json.get('frame'))
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run detection
        results = model(img)

        objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                name = model.names[cls]
                if name == "Bottle":
                    objects.append({
                        'name': name,
                        'confidence': conf,
                        'xmin': int(x1),
                        'ymin': int(y1),
                        'xmax': int(x2),
                        'ymax': int(y2)
                    })

        return objects

    except base64.binascii.Error as e:
        # This will catch any remaining padding or bad-character errors
        return jsonify({"error": f"Base64 decoding error: {e}. Check client data."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# Run the app (for development)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

