import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from keypoints_interfaces.msg import PoseKeypoints, PoseKeypointsArray
import mediapipe as mp
import cv2

class PoseDetectionNode(Node):
    def __init__(self):
        super().__init__('pose_detection_node')
        self.bridge = CvBridge()
        self.frame_count = 0

        # Mediapipe Pose Detector
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Subscribe to camera frames
        self.subscription = self.create_subscription(
            Image,
            '/zed/zed_node/left/image_rect_color',
            self.image_callback,
            10
        )

        # Publisher for keypoints
        self.publisher_ = self.create_publisher(PoseKeypointsArray, 'pose_keypoints', 10)
        self.get_logger().info("Pose Detection Node started")

    def image_callback(self, msg):
        # Convert ROS 2 Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        # Publish detected keypoints
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            # cv2.imwrite(f"pose_overlay_{self.frame_count}.jpg", frame)
            self.frame_count += 1
            keypoints_array = PoseKeypointsArray()
            for lm_id, lm in enumerate(results.pose_landmarks.landmark):
                keypoint = PoseKeypoints()
                keypoint.name = self.mp_pose.PoseLandmark(lm_id).name
                keypoint.x = lm.x
                keypoint.y = lm.y
                keypoints_array.keypoints.append(keypoint)
                # self.publisher_.publish(keypoint)
                # self.get_logger().info(f"Published keypoint: {keypoint.name} ({keypoint.x}, {keypoint.y})")
            self.publisher_.publish(keypoints_array)
            self.get_logger().info(f"Published keypoints for frame: {self.frame_count}")


def main(args=None):
    rclpy.init(args=args)
    pose_detection_node = PoseDetectionNode()
    try:
        rclpy.spin(pose_detection_node)
    except KeyboardInterrupt:
        pose_detection_node.get_logger().info("Shutting down Pose Detection Node")
    finally:
        pose_detection_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
