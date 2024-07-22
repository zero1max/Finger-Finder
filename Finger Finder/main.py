import cv2
import mediapipe as mp
import webbrowser

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_drawing = mp.solutions.drawing_utils

def detect_two_fingers(hand_landmarks):
    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark

        index_finger_tip = landmarks[8]
        middle_finger_tip = landmarks[12]
        ring_finger_tip = landmarks[16]
        pinky_tip = landmarks[20]

        threshold_y = (landmarks[0].y + landmarks[9].y) / 2  

        index_up = index_finger_tip.y < threshold_y
        middle_up = middle_finger_tip.y < threshold_y
        ring_down = ring_finger_tip.y > threshold_y
        pinky_down = pinky_tip.y > threshold_y

        return index_up and middle_up and ring_down and pinky_down
    return False

cap = cv2.VideoCapture(0)

print("Ikki barmoq ishorasini izlash...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if result.multi_hand_landmarks and detect_two_fingers(result.multi_hand_landmarks):
            print("Two-finger gesture detected!")
            webbrowser.open("https://www.instagram.com/zero.1.max/")
            break

        cv2.imshow("zero1max", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()
hands.close()
