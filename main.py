from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Configure Gemini
genai.configure(api_key='AIzaSyB-lr7H6EWZh2uCgaEkC1UyVtQrbcDx04s')

# Use the correct model name for free tier
model = genai.GenerativeModel('gemini-2.5-flash')


@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        data = request.json
        image_base64 = data['frame']
        return jsonify({
            'description': image_base64,
            'objects': []
        })

        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        # Decode base64 to image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))

        # Create prompt for detailed detection
        prompt = """You are an AI assistant for blind people. Analyze this image and provide a brief, essential description.

        PRIORITY WARNINGS (if present):
        - Danger: stairs, traffic, obstacles, hazards
        - Warning signs or alerts

        OBJECTS (be specific):
        - Brands: Include key details (e.g., "Coca-Cola Zero Sugar" not just "Coca-Cola")
        - Medications/Drugs: Full name and dosage (e.g., "Ibuprofen 200mg tablets")
        - Playing cards: Exact card (e.g., "Ace of Spades")
        - Food/Drinks: Include important details (sugar-free, diet, caffeine-free, etc.)

        FORMAT: 
        - Keep response under 15 words
        - Most important information first
        - Be precise and clear
        - Only mention what's clearly visible

        Example good responses:
        "Danger: stairs ahead"
        "Coca-Cola Zero Sugar can, smartphone on table"
        "Aspirin 500mg bottle, glass of water"
        "Traffic light red, crosswalk ahead"
        "Playing card: King of Hearts"
        """

        # Generate response
        response = model.generate_content([prompt, image])

        return jsonify({
            'description': response.text,
            'objects': []
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e), 'objects': []}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)