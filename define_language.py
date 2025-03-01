import cv2
import mediapipe as mp
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def is_vulcan_salute(landmarks):
    index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    middle_tip = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    ring_tip = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
    pinky_tip = np.array([landmarks[20].x, landmarks[20].y, landmarks[20].z])
    distance_index_middle = np.linalg.norm(index_tip - middle_tip)
    distance_middle_ring = np.linalg.norm(middle_tip - ring_tip)
    distance_ring_pinky = np.linalg.norm(ring_tip - pinky_tip)
    return 1.2 * distance_index_middle < distance_middle_ring and 1.2 * distance_ring_pinky < distance_middle_ring

def calculate_center(landmarks):
    l_shoulder = landmarks.get("LEFT_SHOULDER")
    r_shoulder = landmarks.get("RIGHT_SHOULDER")
    l_hip = landmarks.get("LEFT_HIP")
    r_hip = landmarks.get("RIGHT_HIP")

    if l_shoulder and r_shoulder and l_hip and r_hip:
        center_x = (l_shoulder["x"] + r_shoulder["x"] + l_hip["x"] + r_hip["x"]) / 4
        center_y = (l_shoulder["y"] + r_shoulder["y"] + l_hip["y"] + r_hip["y"]) / 4
        return center_x, center_y
    else:
        return None, None

def make_relative_to_center(landmarks, center_x, center_y):
    normalized_landmarks = {}
    for key, coords in landmarks.items():
        normalized_landmarks[key] = {
            "x": coords["x"] - center_x,
            "y": coords["y"] - center_y
        }
    return normalized_landmarks

def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )
            keypoints = {mp_pose.PoseLandmark(lm_id).name: {"x": lm.x, "y": lm.y} 
                         for lm_id, lm in enumerate(results.pose_landmarks.landmark)}
            keypoints_list.append(keypoints)
    cap.release()
    return keypoints_list

def calculate_salient_keypoints(keypoints_list, movement_threshold=0.1, max_keypoints=3):
    keypoint_groups = {
        "RIGHT_HAND": ["RIGHT_WRIST", "RIGHT_PINKY", "RIGHT_THUMB"],
        "LEFT_HAND": ["LEFT_WRIST", "LEFT_PINKY", "LEFT_THUMB"],
        "FACE": ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT"]
    }
    group_reps = {
        "RIGHT_HAND": "RIGHT_WRIST",
        "LEFT_HAND": "LEFT_WRIST",
        "FACE": "NOSE"
    }
    keypoint_movement = {}
    for keypoints in keypoints_list:
        for key, coord in keypoints.items():
            if key not in keypoint_movement:
                keypoint_movement[key] = {"x": [], "y": []}
            keypoint_movement[key]["x"].append(coord["x"])
            keypoint_movement[key]["y"].append(coord["y"])

    movement_scores = {}
    for key, coords in keypoint_movement.items():
        x_movement = np.ptp(coords["x"])
        y_movement = np.ptp(coords["y"])
        total_movement = x_movement + y_movement
        movement_scores[key] = total_movement

    salient_keypoints = []
    for group, members in keypoint_groups.items():
        max_movement = 0
        selected_keypoint = None
        for key in members:
            if key in movement_scores and movement_scores[key] > max_movement:
                max_movement = movement_scores[key]
                selected_keypoint = key
        if selected_keypoint and max_movement > movement_threshold:
            salient_keypoints.append((selected_keypoint, max_movement))
    salient_keypoints.sort(key=lambda x: x[1], reverse=True)
    salient_keypoints = [key for key, movement in salient_keypoints[:max_keypoints]]

    return salient_keypoints

def simplify_with_rdp(image, epsilon=5.0):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    simplified_image = np.zeros_like(image)
    for i, contour in enumerate(contours):
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(simplified_image, [simplified_contour], -1, (255), thickness=1)
    return simplified_image

def plot_keypoints_image_opencv(keypoints_list, salient_keypoints, img_size=(224, 224)):
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
    x_coords = []
    y_coords = []
    for frame_keypoints in keypoints_list:
        for key in salient_keypoints:
            if key in frame_keypoints:
                x_coords.append(frame_keypoints[key]['x'])
                y_coords.append(frame_keypoints[key]['y'])

    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    for frame_idx in range(len(keypoints_list) - 1):
        frame_keypoints_1 = keypoints_list[frame_idx]
        frame_keypoints_2 = keypoints_list[frame_idx + 1]
        for key in salient_keypoints:
            if key in frame_keypoints_1 and key in frame_keypoints_2:
                x1 = int((frame_keypoints_1[key]["x"] - centroid_x + 0.5) * img_size[0])  # Centering and scaling
                y1 = int((frame_keypoints_1[key]["y"] - centroid_y + 0.5) * img_size[1])
                x2 = int((frame_keypoints_2[key]["x"] - centroid_x + 0.5) * img_size[0])
                y2 = int((frame_keypoints_2[key]["y"] - centroid_y + 0.5) * img_size[1])
                cv2.line(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)

    simplified_img = simplify_with_rdp(img)

    return simplified_img

def save_language(language, save_path):
    with open(save_path, "w") as f:
        json.dump(language, f, indent=4)
    print(f"Language saved to {save_path}")


# Main Program
def main():
    print("This program allows you to define a set of gestures for your custom language.\n")

    # Ask user for the number of gestures
    num_gestures = int(input("Enter the number of gestures you want to define: "))
    language = {}

    save_path = input("Enter the path to save the language file (e.g., gestures.json): ")
    if os.path.exists(save_path):
        append_choice = input(f"The file '{save_path}' already exists. Do you want to append to it? (y/n): ").strip().lower()
        if append_choice == 'y':
            with open(save_path, 'r') as f:
                try:
                    language = json.load(f)
                    print(f"Loaded existing gesture language from {save_path}")
                except:
                    print("Error: The existing file is not a valid JSON. Creating a new file instead.")
                    language = {}
        else:
            print("Creating a new gesture language file.")

    # Mediapipe Pose Setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    def prioritize_hand_by_bounding_box(hand_results):
        def get_bounding_box_size(hand_landmarks):
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            return (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        bounding_boxes = [
            (i, get_bounding_box_size(hand))
            for i, hand in enumerate(hand_results.multi_hand_landmarks)
        ]
        largest_hand_index = max(bounding_boxes, key=lambda x: x[1])[0]
        prioritized_hand = hand_results.multi_hand_landmarks[largest_hand_index]
        label = hand_results.multi_handedness[largest_hand_index].classification[0].label
        return hand_results.multi_hand_landmarks[largest_hand_index], label

    for i in range(num_gestures):
        print(f"\nGesture {i + 1}:")

        # Ask user to record a video or provide an existing video path
        choice = input("Do you want to record a new video with start and stop poses (y/n)? ").lower()
        if choice == "y":
            print("Recording will start when the start pose is detected. Perform the stop pose to end recording.")
            cap = cv2.VideoCapture(0)

            collected_keypoints = []
            state = "IDLE"
            start_pose_frames = 0
            stop_pose_frames = 0
            START_POSE_THRESHOLD = 30
            STOP_POSE_THRESHOLD = 30

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                hand_results = hands.process(image_rgb)

                if results.pose_landmarks:
                    if hand_results.multi_hand_landmarks:
                        prioritized_hand, label = prioritize_hand_by_bounding_box(hand_results)
                        landmarks = {i: lm for i, lm in enumerate(prioritized_hand.landmark)}
                        vulcan_salute_detected = is_vulcan_salute(landmarks)
                        if vulcan_salute_detected:
                            start_pose_frames += 1
                            stop_pose_frames += 1
                        else:
                            start_pose_frames = 0
                            stop_pose_frames = 0

                    if state == "IDLE":
                        if start_pose_frames >= START_POSE_THRESHOLD:
                            state = "RECORDING"
                            start_pose_frames = 0
                            collected_keypoints = []
                            print("Recording started...")
                            stop_pose_frames = 0

                    elif state == "RECORDING":
                        # Record keypoints during the gesture
                        frame_keypoints = {
                                mp_pose.PoseLandmark(lm_id).name: {"x": lm.x, "y": lm.y}
                                for lm_id, lm in enumerate(results.pose_landmarks.landmark)
                                }
                        collected_keypoints.append(frame_keypoints)

                        if stop_pose_frames >= STOP_POSE_THRESHOLD:
                            state = "IDLE"
                            print("Recording stopped.")
                            break

                # Show the camera feed
                cv2.imshow("Recording Gesture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Recording canceled.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            cap.release()
            cv2.destroyAllWindows()

            keypoints_list = collected_keypoints

        else:
            video_path = input("Enter the path to the pre-recorded video: ")
            print("Processing the video to extract keypoints...")
            keypoints_list = extract_keypoints_from_video(video_path)

        salient_keypoints = calculate_salient_keypoints(keypoints_list)
        print(f"Salient keypoints identified: {salient_keypoints}")

        print("Generating a visualization of the gesture...")
        image_array = plot_keypoints_image_opencv(keypoints_list, salient_keypoints)

        gesture_name = input(f"Enter a label for gesture {i + 1}: ")

        language[gesture_name] = {
                "image": image_array.tolist(),
                "salient_keypoints": salient_keypoints,
                }

        print(f"Gesture '{gesture_name}' added to the language.")

    save_language(language, save_path)
    print("Gesture language definition completed!")

if __name__ == "__main__":
    main()
