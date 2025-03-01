import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from keypoints_interfaces.msg import HandKeypoints, HandKeypointsArray
from std_msgs.msg import Bool
import mediapipe as mp
import cv2
import numpy as np

class HandDetectionNode(Node):
    def __init__(self):
        super().__init__('hand_detection_node')
        self.bridge = CvBridge()
        self.frame_count = 0

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

        self.subscription = self.create_subscription(
            Image,
            '/zed/zed_node/left/image_rect_color',
            self.image_callback,
            10
        )

        self.keypoints_publisher = self.create_publisher(HandKeypointsArray, 'hand_keypoints', 10)
        self.vulcan_salute_publisher = self.create_publisher(Bool, 'vulcan_salute_detected', 10)
        self.get_logger().info("Hand Detection Node started")

    def is_vulcan_salute(self, hand_landmarks):
        try:
            index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y, hand_landmarks[8].z])
            middle_tip = np.array([hand_landmarks[12].x, hand_landmarks[12].y, hand_landmarks[12].z])
            ring_tip = np.array([hand_landmarks[16].x, hand_landmarks[16].y, hand_landmarks[16].z])
            pinky_tip = np.array([hand_landmarks[20].x, hand_landmarks[20].y, hand_landmarks[20].z])

            distance_index_middle = np.linalg.norm(index_tip - middle_tip)
            distance_middle_ring = np.linalg.norm(middle_tip - ring_tip)
            distance_ring_pinky = np.linalg.norm(ring_tip - pinky_tip)

            return 1.2 * distance_index_middle < distance_middle_ring and 1.2 * distance_ring_pinky < distance_middle_ring
        except IndexError:
            return False

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            def get_bounding_box_size(hand_landmarks):
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                return (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
            bounding_boxes = [
                (i, get_bounding_box_size(hand))
                for i, hand in enumerate(results.multi_hand_landmarks)
            ]
            largest_hand_index = max(bounding_boxes, key=lambda x: x[1])[0]
            closest_hand_landmarks = results.multi_hand_landmarks[largest_hand_index]
            landmarks = {i: lm for i, lm in enumerate(closest_hand_landmarks.landmark)}
            vulcan_salute_detected = self.is_vulcan_salute(landmarks)
            self.vulcan_salute_publisher.publish(Bool(data=bool(vulcan_salute_detected)))
            keypoints_array = HandKeypointsArray()
            for lm_id, lm in enumerate(closest_hand_landmarks.landmark):
                keypoint = HandKeypoints()
                keypoint.name = self.mp_hands.HandLandmark(lm_id).name
                keypoint.x = lm.x
                keypoint.y = lm.y
                keypoint.z = lm.z
                keypoints_array.keypoints.append(keypoint)
            self.keypoints_publisher.publish(keypoints_array)
            self.frame_count += 1
            self.get_logger().info(f"Processed frame {self.frame_count}, Vulcan Salute: {vulcan_salute_detected}")


def main(args=None):
    rclpy.init(args=args)
    hand_detection_node = HandDetectionNode()
    try:
        rclpy.spin(hand_detection_node)
    except KeyboardInterrupt:
        hand_detection_node.get_logger().info("Shutting down Hand Detection Node")
    finally:
        hand_detection_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
