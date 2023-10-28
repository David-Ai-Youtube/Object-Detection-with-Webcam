# Webcam Object Detection Flask App
Webcam Object Detection Flask App

![Example Detection](https://github.com/David-Ai-Youtube/Object-Detection-with-Webcam/blob/main/example.png?raw=true)


A Flask-based web application for real-time object detection using a webcam. This app integrates a DETR (Detection Transformer) model from Facebook for efficient and accurate object detection in uploaded images. It provides a user-friendly interface where users can upload images, and the detected objects are highlighted and labeled directly in the browser. The app is built using Flask, PIL for image processing, and the transformers library for implementing the machine learning model.

Webcam Object Detection Flask App
Introduction

This repository contains a Flask-based web application that enables real-time object detection through a webcam interface. Utilizing the DETR (Detection Transformer) model from Facebook, the application can detect and label objects in uploaded images. This project demonstrates the integration of deep learning models with web technologies for practical applications.
Features

    Real-time Object Detection: Upload images and see object detection results in real-time.
    DETR Model Integration: Leverages the powerful DETR model for accurate and efficient object detection.
    User-Friendly Interface: Simple and intuitive web interface for interacting with the object detection model.

Technologies Used

    Flask: For the web server and handling HTTP requests.
    Transformers Library: For loading and using the DETR model.
    PIL (Python Imaging Library): For image processing tasks.
    HTML/CSS: For the frontend user interface.

Installation & Setup

    Clone the Repository

    sh
    Copy code
    git clone https://github.com/David-Ai-Youtube/Object-Detection-with-Webcam
    cd Object-Detection-with-Webcam

    Install Dependencies

    sh
    Copy code
    pip install -r requirements.txt

    Run the Application

    sh
    Copy code
    python app.py

    Navigate to http://127.0.0.1:5000/ in your web browser to use the application.

Usage

    Start the Flask server.
    Open the provided URL in a web browser.
    Use the interface to upload images and view the object detection results.

Contributing

Contributions to this project are welcome! Please feel free to submit pull requests, report bugs, and suggest new features or improvements.
License

MIT License - feel free to use this project as you wish.
Acknowledgments

    DETR model by Facebook AI.
    Flask community for an excellent micro web framework.


