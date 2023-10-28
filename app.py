from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='uploads')
app.config['UPLOAD_FOLDER'] = 'uploads'  # Make sure this directory exists
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16 MB file


# Load model and processor
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def handle_webcam_image():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image and detect objects
        processed_filename = detect(filepath)  # Assuming detect returns the filename of processed image

        # Send back the URL or path of the processed image
        return jsonify({"processedImage": url_for('static', filename='webcam_processed.jpg')})
    return jsonify({"error": "No file received"})

def detect(filepath):
    # Open the image file
    image = Image.open(filepath)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Set the target size for post-processing
    # Ensure it's a list of tuples, each tuple representing (height, width) of an image
    target_sizes = [(image.size[1], image.size[0])]  # image.size returns (width, height)

    # Decode the model's predictions
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)
    scores = results[0]['scores'].detach().numpy()
    labels = results[0]['labels'].detach().numpy()

    # Further processing, such as drawing bounding boxes, filtering results, etc.
    # Example (you might need to adapt this part):

    draw = ImageDraw.Draw(image)
    for score, label, box in zip(scores, labels, results[0]['boxes']):
        if score > 0.5:  # Threshold can be adjusted
            box = [round(i.item()) for i in box]  # Convert tensor elements to integers
            draw.rectangle(box, outline="red")
            draw.text((box[0], box[1]), f'{model.config.id2label[label]}: {score:.2f}', fill="red")


    # Save or return the processed image
    processed_filepath = filepath.replace('.jpg', '_processed.jpg')  # Adapt the extension as needed
    image.save(processed_filepath)

    return processed_filepath

if __name__ == "__main__":
    app.run(debug=True)
