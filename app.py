import cv2 as cv
import numpy as np
import mediapipe as mp
import copy
import itertools
import csv
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from utils.cvfpscalc import CvFpsCalc

# --- Flask App ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-super-secret-key-you-should-change!'
socketio = SocketIO(app)

# --- Load AI models ---
keypoint_classifier = KeyPointClassifier()
with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
cvFpsCalc = CvFpsCalc(buffer_len=10)

# --- Sentence logic ---
sentence = ""
last_prediction = ""
prediction_count = 0
PREDICTION_THRESHOLD = 20

# --- Flask route ---
@app.route('/')
def index():
    return render_template('index.html')

# --- SocketIO: process video frames ---
@socketio.on('video_frame')
def process_video_frame(data):
    global sentence, last_prediction, prediction_count

    # Decode image
    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    image = cv.flip(image, 1)

    fps = cvFpsCalc.get()
    debug_image = copy.deepcopy(image)

    # --- AI hand detection ---
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    current_prediction = ""
    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            current_prediction = keypoint_classifier_labels[hand_sign_id]

            # Draw landmarks only
            debug_image = draw_landmarks(debug_image, landmark_list)
            debug_image = draw_info_text(debug_image, brect, handedness, current_prediction)

    # --- Sentence logic ---
    if current_prediction != "" and current_prediction == last_prediction:
        prediction_count += 1
    else:
        last_prediction = current_prediction
        prediction_count = 0

    if prediction_count > PREDICTION_THRESHOLD:
        if len(sentence) == 0 or (len(sentence) > 0 and sentence[-1] != current_prediction):
            sentence += current_prediction
        prediction_count = 0

    # Draw FPS only
    debug_image = draw_info(debug_image, fps)

    # Encode image
    _, buffer = cv.imencode('.jpg', debug_image)
    processed_image_data = base64.b64encode(buffer).decode('utf-8')

    # Emit processed frame + sentence
    emit('processed_frame', {
        'image': 'data:image/jpeg;base64,' + processed_image_data,
        'sentence': sentence
    })

# --- Handle key events (space/backspace) ---
@socketio.on('key_event')
def handle_key_event(json):
    global sentence
    key = json.get('data')
    if key == ' ':
        sentence += " "
    elif key == 'Backspace':
        sentence = sentence[:-1]

# --- Helper functions ---
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 1
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Draw lines and keypoints (same as before)
        # [Omitted here for brevity – keep your original landmark drawing code]
        for index, landmark in enumerate(landmark_point):
            if index in [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255,255,255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0,0,0), 1)
            if index in [4,8,12,16,20]:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255,255,255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0,0,0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label
    if hand_sign_text != "":
        info_text += ":" + hand_sign_text
    cv.putText(image, info_text, (brect[0]+5, brect[1]-4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
    return image

def draw_info(image, fps):
    cv.putText(image, f"FPS:{int(fps)}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv.LINE_AA)
    cv.putText(image, f"FPS:{int(fps)}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)
    return image

# --- Run App ---
if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
