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
        m = moments(image)
        hu = moments_hu(m)
        return np.log(np.abs(hu) + 1e-7)  # log scale to hu moments

    def compute_zernike_moments(self, image, radius=21, degree=8):
        if len(image.shape) == 3:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (radius * 2, radius * 2))
        zernike = mahotas.features.zernike_moments(image, radius=radius, degree=degree)
        return np.array(zernike)

    """def compute_combined_features(self, image):
        hu = self.compute_hu_moments(image)
        zernike = self.compute_zernike_moments(image)
        return np.concatenate((hu, zernike))"""  # do NOT concatenate. Instead separate them, and use softmax difference to classify

    def get_language_features(self):
        features = {}
        for gesture, info in self.language.items():
            img = np.array(info['image'])
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img = cv2.GaussianBlur(img, (7, 7), 0)
            features[gesture] = {
                "gesture": gesture,
                "hu": self.compute_hu_moments(img),
                "zernike": self.compute_zernike_moments(img)
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
        """best_match = None
        min_distance = float('inf')
        for gesture, info in self.language.items():
            salient_keypoints = info['salient_keypoints']
            image_array = self.plot_keypoints_to_image_array(current_keypoints, salient_keypoints)
            image_array = cv2.GaussianBlur(image_array, (7, 7), 0)
            cv2.imwrite(f"{gesture}.png", image_array)
            # compare_to = self.language_histograms[gesture]
            # diff = np.linalg.norm(compare_to - self.compute_bow_histogram(image_array, self.kmeans, self.num_clusters))
            current_features = self.compute_combined_features(image_array)
            diff = np.linalg.norm(self.language_features[gesture] - current_features)
            if diff < min_distance:
                min_distance = diff
                best_match = gesture
        return best_match"""
        hu_diffs = []
        zernike_diffs = []
        for gesture, features in self.language_features.items():
            salient_keypoints = self.language[gesture]['salient_keypoints']
            image_array = self.plot_keypoints_to_image_array(current_keypoints, salient_keypoints)
            image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            image_array = cv2.GaussianBlur(image_array, (7, 7), 0)
            cv2.imwrite(f"{gesture}.png", image_array)
            current_hu = self.compute_hu_moments(image_array)
            current_zernike = self.compute_zernike_moments(image_array)
            hu_diff = np.linalg.norm(current_hu - features['hu'])
            zernike_diff = np.linalg.norm(current_zernike - features['zernike'])

            hu_diffs.append(hu_diff)
            zernike_diffs.append(zernike_diff)
        hu_probs = softmax(-np.array(hu_diffs))
        scaled_zernike = -np.log(np.array(zernike_diffs) + 1e-7)
        zernike_probs = softmax(-np.array(scaled_zernike))
        
        print(f"probs: hu={hu_probs}, zernike={zernike_probs}")
        hu_vote = np.argmax(hu_probs)
        zernike_vote = np.argmax(zernike_probs)

        if hu_vote == zernike_vote:
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
                return list(self.language_features.keys())[zernike_vote]

def main():
    recognizer = GestureRecognizer(language_path="barbados_trial.json", num_clusters=10, random_state=10)
    # recognizer.train_bow()
    recognizer.language_features = recognizer.get_language_features()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    START_POSE_THRESHOLD = 150
    STOP_POSE_THRESHOLD = 150
    state = "IDLE"
    start_pose_frames = 0
    stop_pose_frames = 0
    collected_keypoints = []
    cap = cv2.VideoCapture(0)
    
    def start_pose_detected(pose_landmarks, mp_pose):
        right_wrist_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
        mouth_right_y = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y
        return right_wrist_y < mouth_right_y
    def stop_pose_detected(pose_landmarks, mp_pose):
        right_wrist_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
        mouth_right_y = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y
        return right_wrist_y < mouth_right_y

    while True:
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            if start_pose_detected(results.pose_landmarks, mp_pose):
                start_pose_frames += 1
            else:
                start_pose_frames = 0

            if stop_pose_detected(results.pose_landmarks, mp_pose):
                stop_pose_frames += 1
            else:
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
