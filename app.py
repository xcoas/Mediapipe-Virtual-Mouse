import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
from pynput.mouse import Controller, Button
import pyautogui

class HandLandmarkerClass():
    def __init__(self, num_hands=2):
        self.num_hands = num_hands
        self.num_fingers = num_hands * 5
        self.base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        self.options = vision.HandLandmarkerOptions(
            base_options=self.base_options,
            num_hands=self.num_hands,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(self.options)

        self.camera = cv2.VideoCapture(0)

        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.screen_width, self.screen_height = pyautogui.size()
        self.mouse = Controller()

        self.current_state = ["NEUTRAL"] * self.num_hands

        self.is_moving = [False] * self.num_hands
        self.prev_finger_x = [0.0] * self.num_hands
        self.prev_finger_y = [0.0] * self.num_hands
        
        self.smoothed_delta_x = [0.0] * self.num_hands
        self.smoothed_delta_y = [0.0] * self.num_hands

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),       
            (0, 5), (5, 6), (6, 7), (7, 8),       
            (5, 9), (9, 10), (10, 11), (11, 12),  
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)                               
        ]

        self.FINGERS_TO_CHECK = {
            'kciuk':      (4, 2, 17),
            'wskazujacy': (8, 6, 0),
            'srodkowy':   (12, 10, 0),
            'serdeczny':  (16, 14, 0),
            'maly':       (20, 18, 0) 
        }

    def draw_hands(self, frame, detection_result):
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for connection in self.HAND_CONNECTIONS:
                    pt1 = hand_landmarks[connection[0]]
                    pt2 = hand_landmarks[connection[1]]

                    cx1, cy1 = int(pt1.x * self.frame_width), int(pt1.y * self.frame_height)
                    cx2, cy2 = int(pt2.x * self.frame_width), int(pt2.y * self.frame_height)
                    
                    cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 5)

                for landmark in hand_landmarks:
                    cx = int(landmark.x * self.frame_width)
                    cy = int(landmark.y * self.frame_height)
                    cv2.circle(frame, (cx, cy), 7, (0, 255, 0), -1)

    def return_fingers(self, detection_result):
        if detection_result.hand_landmarks:
            hands_list = []
        
            for hand_landmarks in detection_result.hand_landmarks:
                fingers_list = []
                for finger_name, (top, middle, ref) in self.FINGERS_TO_CHECK.items():
                    
                    pt_top = hand_landmarks[top]
                    pt_middle = hand_landmarks[middle]
                    pt_ref = hand_landmarks[ref]
                    
                    x_top, y_top = int(pt_top.x * self.frame_width), int(pt_top.y * self.frame_height)
                    x_middle, y_middle = int(pt_middle.x * self.frame_width), int(pt_middle.y * self.frame_height)
                    x_ref, y_ref = int(pt_ref.x * self.frame_width), int(pt_ref.y * self.frame_height)

                    dist_top = math.hypot(x_top - x_ref, y_top - y_ref)
                    dist_middle = math.hypot(x_middle - x_ref, y_middle - y_ref)

                    fingers_list.append({
                        'name': finger_name,
                        'top': pt_top, 
                        'middle': pt_middle, 
                        'ref': pt_ref, 
                        'is_showed': dist_top > dist_middle
                    })

                hands_list.append(fingers_list)
            return hands_list
        return []

    def get_gesture(self, hands_list):
        for i, hand in enumerate(hands_list):
            thumb = hand[0]
            pointer = hand[1]
            middle = hand[2]

            tx, ty, tz = thumb['top'].x, thumb['top'].y, thumb['top'].z
            px, py, pz = pointer['top'].x, pointer['top'].y, pointer['top'].z
            mx, my, mz = middle['top'].x, middle['top'].y, middle['top'].z

            pmx, pmy, pmz = pointer['middle'].x, pointer['middle'].y, pointer['middle'].z
            mmx, mmy, mmz = middle['middle'].x, middle['middle'].y, middle['middle'].z

            click_dist_threshold = 0.05
            drop_dist_threshold = 0.08
            far_dist_threshold = 0.12

            click_dist = math.sqrt((px - tx)**2 + (py - ty)**2 + (pz - tz)**2)
            double_click_dist = math.sqrt((mx - tx)**2 + (my - ty)**2 + (mz - tz)**2)

            scroll_dist_top = math.sqrt((px - mx)**2 + (py - my)**2 + (pz - mz)**2)
            scroll_dist_mid = math.sqrt((pmx - mmx)**2 + (pmy - mmy)**2 + (pmz - mmz)**2)

            state = self.current_state[i]

            # ---------------------------------------------------------
            # 1. ZWALNIANIE AKCJI (Wyjście ze stanu)
            # ---------------------------------------------------------
            if state == "DRAG":
                if click_dist > drop_dist_threshold or double_click_dist > drop_dist_threshold:
                    self.current_state[i] = "COOLDOWN"
                    self.mouse.release(Button.left)
            
            elif state == "CLICK":
                if click_dist > drop_dist_threshold:
                    self.current_state[i] = "COOLDOWN"
            
            elif state == "DOUBLE_CLICK":
                if double_click_dist > drop_dist_threshold:
                    self.current_state[i] = "COOLDOWN"

            elif state == "SCROLL":
                if scroll_dist_top > drop_dist_threshold or scroll_dist_mid > drop_dist_threshold or hand[0]['is_showed'] or hand[3]['is_showed'] or hand[4]['is_showed']:
                    self.current_state[i] = "COOLDOWN"
            
            elif state == "COOLDOWN":
                if click_dist > drop_dist_threshold and double_click_dist > drop_dist_threshold:
                    self.current_state[i] = "NEUTRAL"


            # ---------------------------------------------------------
            # 2. ROZPOCZYNANIE AKCJI (Tylko ze stanu NEUTRAL)
            # ---------------------------------------------------------
            if self.current_state[i] == "NEUTRAL":
                if click_dist < click_dist_threshold and double_click_dist < click_dist_threshold and pointer['is_showed'] and middle['is_showed']:
                    self.current_state[i] = "DRAG"
                    self.mouse.press(Button.left)
                
                elif click_dist < click_dist_threshold and double_click_dist > far_dist_threshold and pointer['is_showed']:
                    self.current_state[i] = "CLICK"
                    self.mouse.click(Button.left, 1)
                
                elif double_click_dist < click_dist_threshold and click_dist > far_dist_threshold and middle['is_showed']:
                    self.current_state[i] = "DOUBLE_CLICK"
                    self.mouse.click(Button.left, 2)

                elif scroll_dist_top < click_dist_threshold and scroll_dist_mid < click_dist_threshold and not hand[0]['is_showed'] and not hand[3]['is_showed'] and not hand[4]['is_showed']:
                    self.current_state[i] = "SCROLL"
                    self.prev_finger_y[i] = pointer['ref'].y

            # ---------------------------------------------------------
            # 3. RUCH MYSZKĄ
            # ---------------------------------------------------------
            is_moving_gesture = (hand[0]['is_showed'] and pointer['is_showed'] and hand[2]['is_showed'] and hand[3]['is_showed'] and hand[4]['is_showed'])

            if self.current_state[i] == "SCROLL":
                raw_delta_y = self.prev_finger_y[i] - pointer['ref'].y
                
                if abs(raw_delta_y) > 0.02:
                    scroll_ticks = int(raw_delta_y * 100)
                    if scroll_ticks != 0:
                        self.mouse.scroll(0, scroll_ticks)
                        self.prev_finger_y[i] = pointer['ref'].y

            elif is_moving_gesture or self.current_state[i] == "DRAG":
                if not self.is_moving[i]:
                    self.is_moving[i] = True
                    self.prev_finger_x[i] = pointer['ref'].x
                    self.prev_finger_y[i] = pointer['ref'].y
                    self.smoothed_delta_x[i] = 0.0
                    self.smoothed_delta_y[i] = 0.0
                    continue

                raw_delta_x = pointer['ref'].x - self.prev_finger_x[i]
                raw_delta_y = pointer['ref'].y - self.prev_finger_y[i]

                deadzone = 0.001
                if abs(raw_delta_x) < deadzone: raw_delta_x = 0
                if abs(raw_delta_y) < deadzone: raw_delta_y = 0

                smooth_factor = 0.3 
                
                self.smoothed_delta_x[i] = (raw_delta_x * smooth_factor) + (self.smoothed_delta_x[i] * (1.0 - smooth_factor))
                self.smoothed_delta_y[i] = (raw_delta_y * smooth_factor) + (self.smoothed_delta_y[i] * (1.0 - smooth_factor))

                current_mouse_x, current_mouse_y = self.mouse.position
                sensitivity = 1.3

                new_x = current_mouse_x + (self.smoothed_delta_x[i] * self.screen_width * sensitivity)
                new_y = current_mouse_y + (self.smoothed_delta_y[i] * self.screen_height * sensitivity)

                self.mouse.position = (int(new_x), int(new_y))

                self.prev_finger_x[i] = pointer['ref'].x
                self.prev_finger_y[i] = pointer['ref'].y

            else:
                self.is_moving[i] = False

HandLandmarker = HandLandmarkerClass()

while True:
    ret, frame = HandLandmarker.camera.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamps_ms = int(time.time() * 1000)

    detection_result = HandLandmarker.detector.detect_for_video(mp_image, timestamps_ms)

    HandLandmarker.draw_hands(frame, detection_result)
    fingers_list = HandLandmarker.return_fingers(detection_result)
    HandLandmarker.get_gesture(fingers_list)

    cv2.imshow('tracking', frame)

    if cv2.waitKey(1) == ord('q'):
        break

HandLandmarker.camera.release()
cv2.destroyAllWindows()