import cv2
import json
import mediapipe as mp
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.measure import moments, moments_hu
import mahotas
from scipy.special import softmax
import argparse


def is_vulcan_salute(landmarks):
    index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    middle_tip = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    ring_tip = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
    pinky_tip = np.array([landmarks[20].x, landmarks[20].y, landmarks[20].z])
    distance_index_middle = np.linalg.norm(index_tip - middle_tip)
    distance_middle_ring = np.linalg.norm(middle_tip - ring_tip)
    distance_ring_pinky = np.linalg.norm(ring_tip - pinky_tip)
    return 1.2 * distance_index_middle < distance_middle_ring and 1.2 * distance_ring_pinky < distance_middle_ring

class GestureRecognizer:
    def __init__(self, language_path, num_clusters=50, random_state=10):
        with open(language_path, 'r') as f:
            self.language = json.load(f)
        self.kmeans = None
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.language_histograms = {}
        self.language_features = {}

    def extract_descriptors(self, image):
        sift = cv2.SIFT_create(nfeatures=300)
        image = np.array(image, dtype=np.uint8)
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if image.shape == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return descriptors

    def compute_hu_moments(self, image):
        if len(image.shape) == 3:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
        m = moments(image)
        hu = moments_hu(m)
        # return np.log(np.abs(hu) + 1e-7)  # log scale to hu moments
        hu_normalized = (hu - np.mean(hu)) / (np.std(hu) + 1e-7)  # z-score normalization
        return hu_normalized

    def compute_zernike_moments(self, image, radius=21, degree=8):
        if len(image.shape) == 3:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
        image = cv2.resize(image, (radius * 2, radius * 2))
        zernike = mahotas.features.zernike_moments(image, radius=radius, degree=degree)
        zernike = np.array(zernike)
        zernike_normalized = (zernike - np.mean(zernike)) / (np.std(zernike) + 1e-7)  # z-score normalization
        return zernike_normalized

    def process_image_with_rdp(self, image, epsilon=5.0):
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        edges = cv2.Canny(image, threshold1=100, threshold2=200)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        reconstructed_image = np.zeros_like(image)
        for i,contour in enumerate(contours):
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(reconstructed_image, [simplified_contour], -1, (255), thickness=1)
        return reconstructed_image
    
    def compute_fourier_descriptors(self, img, degree=12):
        contour = []
        contour, hierarchy = cv2.findContours(
            img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
            contour)
        contour_array = contour[0][:, 0, :]
        contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
        contour_complex.real = contour_array[:, 0]
        contour_complex.imag = contour_array[:, 1]
        fourier_result = np.fft.fft(contour_complex)
        descriptors = fourier_result
        descriptors = np.fft.fftshift(descriptors)
        center_index = len(descriptors) / 2
        descriptors = descriptors[
            int(center_index - degree / 2):int(center_index + degree / 2)]
        descriptors = np.fft.ifftshift(descriptors)
        return descriptors

    def compute_edge_features(self, image, epsilon=0.01):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(image, threshold1=50, threshold2=150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(10, dtype=np.float32)
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = epsilon * cv2.arcLength(largest_contour, True)
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        simplified_contour = np.squeeze(simplified_contour)
        if len(simplified_contour.shape) != 2:
            return np.zeros(10, dtype=np.float32)
        centroid = np.mean(simplified_contour, axis=0)
        simplified_contour -= centroid
        flattened = simplified_contour.flatten()
        normalized_features = flattened / (np.linalg.norm(flattened) + 1e-7)
        return np.pad(normalized_features, (0, max(0, 10 - len(normalized_features))), mode='constant')

    def get_language_features(self):
        features = {}
        for gesture, info in self.language.items():
            img = np.array(info['image'])
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (7, 7), 0) 
            img = self.process_image_with_rdp(img)
            img = cv2.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=11, C=2)
            features[gesture] = {
                "gesture": gesture,
                "hu": self.compute_hu_moments(img),
                "zernike": self.compute_zernike_moments(img),
                "fd": self.compute_fourier_descriptors(img),
            }
        return features

    def compute_bow_histogram(self, image, kmeans, num_clusters):
        descriptors = self.extract_descriptors(image)
        if descriptors is None:
            return np.zeros(num_clusters)
        word_indices = kmeans.predict(descriptors)
        histogram, _ = np.histogram(word_indices, bins=np.arange(num_clusters + 1))
        histogram = histogram.astype(float)
        histogram /= (histogram.sum() + 1e-7)
        return histogram

    def get_language_histograms(self):
        histograms = {}
        for gesture, info in self.language.items():
            img = info['image']
            histogram = self.compute_bow_histogram(img, self.kmeans, self.num_clusters)
            histograms[gesture] = histogram
        return histograms

    def train_bow(self):
        all_descriptors = []
        for gesture, info in self.language.items():
            img = np.array(info['image'])
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            reference_descriptors = self.extract_descriptors(img)
            if reference_descriptors is not None:
                all_descriptors.extend(reference_descriptors)
        all_descriptors = np.array(all_descriptors)
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        self.kmeans.fit(all_descriptors)
        self.language_histograms = self.get_language_histograms()

    def plot_keypoints_to_image_array(self, keypoints_list, salient_keypoints, img_size=(224, 224)):
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
            kp1 = keypoints_list[frame_idx]
            kp2 = keypoints_list[frame_idx+1]
            for key in salient_keypoints:
                if key in kp1 and key in kp2:
                    x1 = int((kp1[key]['x'] - centroid_x + 0.5) * img_size[0])
                    y1 = int((kp1[key]['y'] - centroid_y + 0.5) * img_size[1])
                    x2 = int((kp2[key]['x'] - centroid_x + 0.5) * img_size[0])
                    y2 = int((kp2[key]['y'] - centroid_y + 0.5) * img_size[1])

                    cv2.line(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
        return img

    def identify_gesture(self, current_keypoints):
        hu_diffs = []
        zernike_diffs = []
        fd_diffs = []
        canny_diffs = []
        for gesture, features in self.language_features.items():
            salient_keypoints = self.language[gesture]['salient_keypoints']
            image_array = self.plot_keypoints_to_image_array(current_keypoints, salient_keypoints)
            image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            image_array = cv2.GaussianBlur(image_array, (7, 7), 0)
            image_array = self.process_image_with_rdp(image_array)
            image_array = cv2.adaptiveThreshold(image_array, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=11, C=2)
            cv2.imwrite(f"{gesture}.png", image_array)
            current_hu = self.compute_hu_moments(image_array)
            current_zernike = self.compute_zernike_moments(image_array)
            current_fd = self.compute_fourier_descriptors(image_array)
            # current_canny = self.compute_edge_features(image_array)
            hu_diff = np.linalg.norm(current_hu - features['hu'])
            zernike_diff = np.linalg.norm(current_zernike - features['zernike'])
            fd_diff = np.linalg.norm(current_fd - features['fd'])
            hu_diffs.append(hu_diff)
            zernike_diffs.append(zernike_diff)
            fd_diffs.append(fd_diff)
        # hu_probs = softmax(-np.array(hu_diffs))
        # scaled_zernike = np.array(zernike_diffs)
        # zernike_probs = softmax(-np.array(scaled_zernike))
        
        # print(f"probs: hu={hu_probs}, zernike={zernike_probs}")
        hu_vote = np.argmin(hu_diffs)
        zernike_vote = np.argmin(zernike_diffs)
        fd_vote = np.argmin(fd_diffs)

        hu_conf = 1 / (hu_diffs[hu_vote] + 1e-7)
        zernike_conf = 1 / (zernike_diffs[zernike_vote] + 1e-7)
        fd_conf = 1 / (fd_diffs[fd_vote] + 1e-7)
        votes = [hu_vote, zernike_vote, fd_vote]
        print(f"Hu vote: {hu_vote}, confidence: {hu_conf}")
        print(f"Zernike vote: {zernike_vote}, confidence: {zernike_conf}")
        print(f"FD vote: {fd_vote}, confidence: {fd_conf}")
        vote_counts = {}
        for vote in votes:
            if vote in vote_counts:
                vote_counts[vote] += 1
            else:
                vote_counts[vote] = 1

        # Find the gesture with the majority of votes
        majority_vote = max(vote_counts, key=vote_counts.get)

        print(f"Final gesture based on majority vote: {majority_vote}")
        return list(self.language_features.keys())[majority_vote]
        """if hu_vote == zernike_vote:
            res = list(self.language_features.keys())[hu_vote]
            print(f"both hu and zernike voted for {res}")
            return res
        else:
            res1 = list(self.language_features.keys())[zernike_vote]
            res2 = list(self.language_features.keys())[hu_vote]
            print(f"zernike vote: {res1}; hu vote: {res2}")
            if hu_probs[hu_vote] > zernike_probs[zernike_vote]:
                return list(self.language_features.keys())[hu_vote]
            else:
                return list(self.language_features.keys())[zernike_vote]"""

def main():
    parser = argparse.ArgumentParser(description="Real-time gesture recognition")
    parser.add_argument("--language", required=True, help="Path to the gesture language JSON file (e.g., gestures.json)")
    args = parser.parse_args()
    recognizer = GestureRecognizer(language_path=args.language, num_clusters=10, random_state=10)
    # recognizer.train_bow()
    recognizer.language_features = recognizer.get_language_features()
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
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

    START_POSE_THRESHOLD = 30
    STOP_POSE_THRESHOLD = 30
    state = "IDLE"
    start_pose_frames = 0
    stop_pose_frames = 0
    collected_keypoints = []
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
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
                    collected_keypoints = []
                    start_pose_frames = 0
                    stop_pose_frames = 0
                    print("recording started...")
            elif state == "RECORDING":
                frame_keypoints = {
                    mp_pose.PoseLandmark(lm_id).name: {"x": lm.x, "y": lm.y}
                    for lm_id, lm in enumerate(results.pose_landmarks.landmark)
                }
                collected_keypoints.append(frame_keypoints)
                
                if stop_pose_frames >= STOP_POSE_THRESHOLD:
                    state = "PROCESSING"
                    stop_pose_frames = 0
                    print("recording stopped. processing...")
            elif state == "PROCESSING":
                recognized_gesture = recognizer.identify_gesture(collected_keypoints)
                print(f"recognized gesture: {recognized_gesture}")
                state = "IDLE"
                start_pose_frames = 0
                stop_pose_frames = 0
        cv2.imshow("gesture recognition", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
