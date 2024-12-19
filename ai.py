import cv2 as cv
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
        static_image_mode = False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
mp_drawing = mp.solutions.drawing_utils
# Global variables for FPS calculation
current_time = 0
previous_time = 0



def detect_hand(hand_predict): 
    global current_time, previous_time  # Use global variables for FPS calculation

    # Prediction process
    hand_predict.flags.writeable = False
    results = holistic_model.process(hand_predict)
    hand_predict.flags.writeable = True
    
    hand_predict = cv.cvtColor(hand_predict, cv.COLOR_RGB2BGR)

    # Draw landmarks
    mp_drawing.draw_landmarks(
        hand_predict,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        hand_predict, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS
    )

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - previous_time) if previous_time > 0 else 0
    previous_time = current_time

    # Return processed frame and FPS details
    fps_text = f"{int(fps)} FPS"
    cv.putText(hand_predict,fps_text, (10, 70), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    return hand_predict



def process_video(): 
    capture = cv.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        processed_frame = detect_hand(frame_rgb)

        # Display the frame
        cv.imshow("Hand Motion Recognition Model", processed_frame)

        # Break on 'q' key
        if cv.waitKey(5) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()

process_video()


