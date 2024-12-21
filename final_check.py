from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)

# Known face encodings and names
known_face_encodings = []
known_face_names = []

# Load known face encodings
def load_known_faces():
    global known_face_encodings, known_face_names

    input1 = face_recognition.load_image_file("11.jpeg")
    ram_face_encoding = face_recognition.face_encodings(input1)[0]

    input2 = face_recognition.load_image_file("gv.jpg")
    gv_face_encoding = face_recognition.face_encodings(input2)[0]

    input3 = face_recognition.load_image_file("megha.jpg")
    megha_face_encoding = face_recognition.face_encodings(input3)[0]

    input4 = face_recognition.load_image_file("jenny.jpg")
    jenny_face_encoding = face_recognition.face_encodings(input4)[0]

    known_face_encodings = [ram_face_encoding, gv_face_encoding, megha_face_encoding, jenny_face_encoding]
    known_face_names = ["ram", "gv", "megha", "jenny"]

# Global variables for streaming
video_capture = None

@app.route('/')
def index():
    return render_template('index.html')

def initialize_video_capture():
    """Start video capture."""
    global video_capture
    if not video_capture or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)

def release_video_capture():
    """Release video capture."""
    global video_capture
    if video_capture and video_capture.isOpened():
        video_capture.release()
        video_capture = None

def generate_frames():
    """Generate frames for the video stream."""
    global video_capture

    while video_capture and video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        # Resize frame for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Perform face recognition
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        # Determine message based on recognized faces
        if "ram" in face_names or "gv" in face_names:
            message = "Bank officer allowed"
        elif "megha" in face_names or "jenny" in face_names:
            message = "Thief not allowed"
        else:
            message = "You are allowed"

        # Display the results on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Add the message to the top of the frame
        cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Stream video feed."""
    initialize_video_capture()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream')
def start_stream():
    """Start the video stream."""
    release_video_capture()
    initialize_video_capture()
    return "Stream started!"

@app.route('/stop_stream')
def stop_stream():
    """Stop the video stream."""
    release_video_capture()
    return "Stream stopped!"

if __name__ == '__main__':
    load_known_faces()
    app.run(debug=True)
