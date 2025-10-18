import base64
from flask import Flask, jsonify, request
from ultralytics import YOLO
import cv2
import numpy as np
import pprint
import sys

# --- 1. Load the model ONCE when the app starts ---
# This is much more efficient.
model = YOLO("yolov8n.pt")

# Create the Flask application instance
app = Flask(__name__)

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
        # --- 1. STRIP DATA URI PREFIX (if it exists) ---
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        # --- 2. FIX: Correctly add padding ---
        padding = (4 - len(frame_data) % 4) % 4
        frame_data_padded = frame_data + ("=" * padding)

        # --- 3. Decode the base64 string ---
        # Note: We decode the *padded* string
        img_bytes = base64.b64decode(frame_data_padded)

        # --- 4. DEBUG: Save the raw bytes to a file ---
        with open("debug_image.jpg", "wb") as f:
            f.write(img_bytes)
        print("DEBUG: Saved received image to 'debug_image.jpg'")

        # --- 5. Check if image is valid by loading it ---
        # We must convert the raw bytes to a NumPy array for cv2
        img_np_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np_array, cv2.IMREAD_COLOR)

        if img is None:
            # The base64 was valid, but the data was NOT an image
            return jsonify({"error": "Failed to decode image from bytes. See 'debug_image.jpg'"}), 400

        img_data = base64.b64decode(frame_data_padded)
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

                objects.append({
                    'name': name,
                    'confidence': conf,
                    'xmin': int(x1),
                    'ymin': int(y1),
                    'xmax': int(x2),
                    'ymax': int(y2)
                })

        return objects
        # return results[0]
        class_names = results[0].names
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = class_names[cls_id]


            detections.append({
                "xmin": float(x1),
                "ymin": float(y1),
                "xmax": float(x2),
                "ymax": float(y2),
                "confidence": float(conf),
                "class_id": cls_id,
                "name": class_name
            })
            print(detections)

        return jsonify({"objects": detections})

    except base64.binascii.Error as e:
        # This will catch any remaining padding or bad-character errors
        return jsonify({"error": f"Base64 decoding error: {e}. Check client data."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# Run the app (for development)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

    import pprint
    import sys

