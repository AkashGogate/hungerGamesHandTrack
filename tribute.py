import math
import cv2
import numpy as np
import mediapipe as mp

# Define a video capture object
vid = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False)

# Drawing specifications
red_dot = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1)
green_line = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

message = 'No signal'

# Convert to RGB outside the loop
with mp_pose.Pose(static_image_mode=True) as pose:
    while True:
        ret, frame = vid.read()
        original = frame

        # Process pose landmarks

        result_pose = pose.process(frame)

        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        result_hands = hand.process(RGB_frame)

        if result_hands.multi_hand_landmarks:
            handLandMarks = result_hands.multi_hand_landmarks[0]

            wrist = handLandMarks.landmark[0]
            thumb = handLandMarks.landmark[4]
            index = handLandMarks.landmark[8]
            ring = handLandMarks.landmark[16]
            pinky = handLandMarks.landmark[20]

            # Calculate distances only if hand landmarks are present
            thumb_ring_distance = math.sqrt((thumb.x - ring.x) ** 2 + (thumb.y - ring.y) ** 2)
            index_pinky_distance = math.sqrt((index.x - pinky.x) ** 2 + (index.y - pinky.y) ** 2)

            # Check if middle 3 fingers are up
            if thumb_ring_distance < 0.16 and index_pinky_distance > 0.145 and wrist.y < 0.6:
                message = 'We have a volunteer as Tribute'
            elif wrist.y < 0.6:
                message = 'Hand is raised'
            else:
                message = 'No Signal'

            mp_draw.draw_landmarks(frame, handLandMarks, mp_hands.HAND_CONNECTIONS)

        mp_draw.draw_landmarks(frame, landmark_list=result_pose.pose_landmarks, landmark_drawing_spec=red_dot)
        mp_draw.draw_landmarks(frame, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=green_line)

        text_image = np.zeros_like(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(text_image, message, (30, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Overlay the text image on the original frame
        frame = cv2.addWeighted(frame, 1, text_image, 1, 0)

        # Use cv2.hconcat for efficient horizontal concatenation
        combined_image = cv2.hconcat([frame, original])

        # Display the combined image
        cv2.imshow('Combined Images', combined_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object
vid.release()
cv2.destroyAllWindows()
