import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

# --- Variables for pinch detection ---
pinch = False
pinch_start = 0
click_threshold = 50

# --- Screenshot Variables ---
last_screenshot_time = 0
screenshot_cooldown = 3.0  # Seconds to wait between screenshots

# --- Initialize webcam ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# --- Initialize hand detector ---
hand_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils

# --- Get screen size ---
screen_width, screen_height = pyautogui.size()

# --- Smoothing and Scroll variables ---
prev_x, prev_y = 0, 0
prev_scroll_y = 0
smoothening = 5
scroll_sensitivity = 40

# --- FPS calculation variables ---
pTime = 0
cTime = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    index_finger = None
    thumb_finger = None
    
    # New variables to track scroll mode state
    middle_tip_y = 0
    middle_knuckle_y = 0

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            # --- Screenshot Detection (Finger Counting) ---
            fingers = []
            # Check 4 fingers (Index, Middle, Ring, Pinky)
            # Tip IDs: 8, 12, 16, 20 | Knuckle IDs: 6, 10, 14, 18
            for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                if landmarks[tip].y < landmarks[pip].y:
                    fingers.append(1) # Finger is open
                else:
                    fingers.append(0) # Finger is closed

            # --- Trigger Screenshot ---
            # If all 4 fingers are closed AND the thumb is closed
            # (Using sum(fingers) == 0 for the 4 main fingers)
            if sum(fingers) == 0:
                current_time = time.time()
                if current_time - last_screenshot_time > screenshot_cooldown:
                    # Take the screenshot
                    ss = pyautogui.screenshot()
                    # Save with a unique timestamp
                    file_name = f"screenshot_{int(current_time)}.png"
                    ss.save(file_name)
                    
                    print(f"Screenshot saved: {file_name}")
                    last_screenshot_time = current_time
                    
                    # Visual feedback on screen
                    cv2.putText(frame, "SCREENSHOT TAKEN!", (150, 250), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, (x, y), 11, (0, 255, 255), -1)

                # --- Track Middle Finger for Mode Switching ---
                if id == 12: # Middle Tip
                    middle_tip_y = y
                    cv2.circle(frame, (x, y), 11, (0, 255, 255), -1)
                if id == 10: # Middle Knuckle
                    middle_knuckle_y = y

                # --- Index Finger Tip ---
                if id == 8:
                    cv2.circle(frame, (x, y), 11, (0, 255, 255), -1)
                    screen_x = screen_width / frame_width * x
                    screen_y = screen_height / frame_height * y

                    # Smooth cursor movement
                    curr_x = prev_x + (screen_x - prev_x) / smoothening
                    curr_y = prev_y + (screen_y - prev_y) / smoothening
                    
                    # Store for pinch logic
                    index_finger = (curr_x, curr_y)

                # --- Thumb Tip ---
                if id == 4:
                    cv2.circle(frame, (x, y), 11, (0, 255, 255), -1)
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y
                    thumb_finger = (thumb_x, thumb_y)    

            # --- MODE LOGIC (Decide between Moving and Scrolling) ---
            # If middle finger is up (tip is higher/smaller y than knuckle), enter Scroll Mode
            if middle_tip_y < middle_knuckle_y and middle_tip_y != 0:
                # SCROLL LOGIC
                if not pinch and prev_scroll_y != 0:
                    diff = screen_y - prev_scroll_y
                    if abs(diff) > 20: # Deadzone to prevent jitter
                        scroll_amount = int(diff / 5) * scroll_sensitivity
                        pyautogui.scroll(-scroll_amount)
                prev_scroll_y = screen_y
            else:
                # CURSOR MOVEMENT MODE
                pyautogui.moveTo(curr_x, curr_y)
                prev_scroll_y = 0 # Reset baseline so it doesn't jump when switching
                
            # Update previous coordinates
            prev_x, prev_y = curr_x, curr_y

               
                

    # --- Pinch detection and click ---
    if index_finger and thumb_finger:
        distance = math.hypot(index_finger[0] - thumb_finger[0],
                              index_finger[1] - thumb_finger[1])
        current_time = time.time()

        if distance < click_threshold:
            pyautogui.click()
            

    # --- FPS calculation and display ---
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
