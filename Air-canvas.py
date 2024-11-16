import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Set up for finger movement tracking
hand_detector = mp.solutions.hands.Hands(max_num_hands=2)  # Allow up to 2 hands
drawing_utils = mp.solutions.drawing_utils

# Color points for drawing
bpoints, gpoints, rpoints, ypoints = [deque(maxlen=1024) for _ in range(4)]
blue_index, green_index, red_index, yellow_index = 0, 0, 0, 0
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Blue, Green, Red, Yellow
colorIndex = 0
brush_thickness = 5
drawing_enabled = True  # Flag to enable/disable drawing
history = []  # Store history for undo feature

# Canvas for drawing
paintWindow = np.ones((800, 1600, 3), dtype=np.uint8) * 255
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

# Smoothing factor
smoothing_factor = 3  # Adjusted smoothing factor

def smooth_points(points):
    smoothed_points = []
    for i in range(1, len(points)):
        valid_points = [p for p in points[max(0, i - smoothing_factor):i + 1] if p is not None]
        if valid_points:
            x = int(np.mean([p[0] for p in valid_points]))
            y = int(np.mean([p[1] for p in valid_points]))
            smoothed_points.append((x, y))
        else:
            smoothed_points.append(None)
    return smoothed_points

def save_canvas_state():
    if len(history) < 1024:  # Limit history to prevent memory overload
        history.append(paintWindow.copy())

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw buttons on both frames
    button_layout = [(60, 1, 210, 65, (122, 122, 122), "CLEAR"),
                     (230, 1, 360, 65, colors[0], ""),   # Blue
                     (390, 1, 520, 65, colors[1], ""),   # Green
                     (550, 1, 680, 65, colors[2], ""),   # Red
                     (710, 1, 840, 65, colors[3], ""),   # Yellow
                     (930, 1, 1050, 65, (200, 200, 200), "THICK +"),
                     (1090, 1, 1210, 65, (200, 200, 200), "THICK -"),
                     (1220, 1, 1350, 65, (122, 122, 122), "UNDO")]
    for x1, y1, x2, y2, color, text in button_layout:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(paintWindow, (x1, y1), (x2, y2), color, -1)  # Draw buttons on paintWindow
        if text:
            cv2.putText(frame, text, (x1 + 20, y1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(paintWindow, text, (x1 + 20, y1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Process the frame for hand detection
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    # Check if two hands are detected for navigation mode
    if hands and len(hands) == 2:
        drawing_enabled = False
    elif hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            # Track index and middle finger positions
            index_finger = (int(landmarks[8].x * frame_width), int(landmarks[8].y * frame_height))
            middle_finger = (int(landmarks[12].x * frame_width), int(landmarks[12].y * frame_height))

            # Calculate the distance between index and middle finger
            distance = np.sqrt((index_finger[0] - middle_finger[0]) ** 2 + (index_finger[1] - middle_finger[1]) ** 2)
            drawing_enabled = distance >= 70  # Increase threshold for larger frame

            for id, landmark in enumerate(landmarks):
                x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)

                if id == 8:  # Index Finger
                    cv2.circle(frame, (x, y), 10, colors[colorIndex], -1)
                    center = (x, y)

                    # Draw donut-shaped circle on paint window as a reference
                    paintWindow_copy = paintWindow.copy()  # Create a copy for reference
                    cv2.circle(paintWindow_copy, center, 15, (0, 0, 0), 2)  # Outer circle
                    cv2.circle(paintWindow_copy, center, 8, (128, 128, 128), -1)  # Inner circle (making it donut-shaped)
                    
                    cv2.imshow("Paint", paintWindow_copy)  # Display the paint window copy

                    if y <= 65:
                        if 60 <= x <= 210:  # Clear Button
                            bpoints, gpoints, rpoints, ypoints = [deque(maxlen=1024) for _ in range(4)]
                            paintWindow[67:, :] = 255
                            history.clear()
                            blue_index = green_index = red_index = yellow_index = 0
                        elif 230 <= x < 360:
                            colorIndex = 0  # Blue
                        elif 390 <= x < 520:
                            colorIndex = 1  # Green
                        elif 550 <= x < 680:
                            colorIndex = 2  # Red
                        elif 710 <= x < 840:
                            colorIndex = 3  # Yellow
                        elif 930 <= x <= 1050:  # Increase thickness
                            brush_thickness = min(20, brush_thickness + 1)
                        elif 1090 <= x <= 1210:  # Decrease thickness
                            brush_thickness = max(1, brush_thickness - 1)
                        elif 1220 <= x <= 1350:  # Undo button
                            if history:
                                paintWindow = history.pop()
                    elif drawing_enabled:
                        save_canvas_state()
                        if colorIndex == 0:
                            if blue_index >= len(bpoints):
                                bpoints.append(deque(maxlen=1024))
                            bpoints[blue_index].appendleft(center)
                        elif colorIndex == 1:
                            if green_index >= len(gpoints):
                                gpoints.append(deque(maxlen=1024))
                            gpoints[green_index].appendleft(center)
                        elif colorIndex == 2:
                            if red_index >= len(rpoints):
                                rpoints.append(deque(maxlen=1024))
                            rpoints[red_index].appendleft(center)
                        elif colorIndex == 3:
                            if yellow_index >= len(ypoints):
                                ypoints.append(deque(maxlen=1024))
                            ypoints[yellow_index].appendleft(center)

    # Draw lines on canvas with smoothing
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            actual_points = list(points[i][j])
            smoothed_points = smooth_points(actual_points)
            for k in range(1, len(smoothed_points)):
                if smoothed_points[k - 1] is None or smoothed_points[k] is None:
                    continue
                # Highlighting the line update
                cv2.line(paintWindow, smoothed_points[k - 1], smoothed_points[k], colors[i], brush_thickness, cv2.LINE_AA)  # Updated to anti-aliased lines
                cv2.line(frame, smoothed_points[k - 1], smoothed_points[k], colors[i], brush_thickness, cv2.LINE_AA)  # Updated to anti-aliased lines

    # Display frames
    cv2.imshow("Air Canvas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
