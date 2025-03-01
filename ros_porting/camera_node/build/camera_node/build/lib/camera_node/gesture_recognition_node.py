import rclpy
from rclpy.node import Node
from keypoints_interfaces.msg import PoseKeypointsArray
from std_msgs.msg import String, Bool
from camera_node.gesture_recognizer import GestureRecognizer  # GestureRecognizer is in a separate file
from ament_index_python.packages import get_package_share_directory
import os
from std_msgs.msg import Float64MultiArray

class GestureRecognitionNode(Node):
    def __init__(self):
        super().__init__('gesture_recognition_node')
        self.state = "IDLE"
        self.start_pose_frames = 0
        self.stop_pose_frames = 0
        self.START_POSE_THRESHOLD = 15
        self.STOP_POSE_THRESHOLD = 15
        self.collected_keypoints = []
        self.last_state = ''
        self.left_screen_msg = String()  # left screen from diver's POV (right from MeCO's POV)
        self.left_screen_msg.data = ''
        self.right_screen_msg = String()
        self.right_screen_msg.data = ''
        package_share_directory = get_package_share_directory('camera_node')
        language_path = os.path.join(package_share_directory, 'resource', 'barbados_trial.json')
        self.recognizer = GestureRecognizer(language_path=language_path, num_clusters=10, random_state=10)
        self.recognizer.language_features = self.recognizer.get_language_features()

        # Subscriber for pose keypoints
        self.subscription = self.create_subscription(
                PoseKeypointsArray,
                '/pose_keypoints',
                self.keypoints_callback,
                10
        )

        self.vulcan_subscription = self.create_subscription(
                Bool,
                '/vulcan_salute_detected',
                self.vulcan_salute_callback,
                10
        )

        # Publisher for recognized gestures
        self.publisher_ = self.create_publisher(String, '/recognized_gesture', 10)

        self.tts_publisher_ = self.create_publisher(String, '/meco/tts', 10)

        self.control_publisher_ = self.create_publisher(Float64MultiArray, '/meco/control', 10)

        self.oled_port_publisher_ = self.create_publisher(String, '/meco/oled_port', 10)

        self.oled_stbd_publisher_ = self.create_publisher(String, '/meco/oled_stbd', 10)

        self.get_logger().info("Gesture Recognition Node started")

    def keypoints_callback(self, msg):
        frame_keypoints = {keypoint.name: {"x": keypoint.x, "y": keypoint.y} for keypoint in msg.keypoints}
        self.process_keypoints(frame_keypoints)

    def vulcan_salute_callback(self, msg):
        if self.state == 'IDLE':
            if self.last_state != 'IDLE':
                state_msg = String()
                state_msg.data = 'entering idle state'
                self.tts_publisher_.publish(state_msg)
            self.left_screen_msg.data = 'State'
            self.right_screen_msg.data = 'IDLE'
            self.oled_stbd_publisher_.publish(self.left_screen_msg)
            self.oled_port_publisher_.publish(self.right_screen_msg)
            self.last_state = 'IDLE'

            if msg.data:
                self.start_pose_frames += 1
                if self.start_pose_frames >= self.START_POSE_THRESHOLD:
                    self.state = 'RECORDING'
                    self.start_pose_frames = 0
                    self.get_logger().info("Recording started...")
            else:
                self.start_pose_frames = 0
        elif self.state == 'RECORDING':
            if msg.data:
                self.stop_pose_frames += 1
                if self.stop_pose_frames >= self.STOP_POSE_THRESHOLD:
                    self.state = 'PROCESSING'
                    self.stop_pose_frames = 0
                    self.get_logger().info('Recording stopped. Processing gesture...')
            else:
                self.stop_pose_frames = 0

    def process_keypoints(self, frame_keypoints):
        if self.state == "RECORDING":
            if self.last_state != 'RECORDING':
                state_msg = String()
                state_msg.data = 'recording gesture'
                self.tts_publisher_.publish(state_msg)
            self.left_screen_msg.data = 'State'
            self.right_screen_msg.data = 'RECORDING'
            self.oled_stbd_publisher_.publish(self.left_screen_msg)
            self.oled_port_publisher_.publish(self.right_screen_msg)
            self.last_state = 'RECORDING'
            self.collected_keypoints.append(frame_keypoints)

        elif self.state == "PROCESSING":
            if self.last_state != 'PROCESSING':
                state_msg = String()
                state_msg.data = 'gesture recorded'
                self.tts_publisher_.publish(state_msg)
            self.last_state = 'PROCESSING'
            recognized_gesture = self.recognizer.identify_gesture(self.collected_keypoints)
            gesture_msg = String()
            gesture_msg.data = recognized_gesture
            self.publisher_.publish(gesture_msg)
            self.tts_publisher_.publish(gesture_msg)
            self.left_screen_msg.data = 'Gesture'
            self.right_screen_msg.data = recognized_gesture[:9]
            self.oled_stbd_publisher_.publish(self.left_screen_msg)
            self.oled_port_publisher_.publish(self.right_screen_msg)
            self.get_logger().info(f"Recognized gesture: {recognized_gesture}")
            movement_msg = Float64MultiArray()
            if recognized_gesture == 'help':
                # move forward
                movement_msg.data = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif recognized_gesture == 'base':
                # descend
                movement_msg.data = [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif recognized_gesture == 'peak':
                # ascend
                movement_msg.data = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                movement_msg.data = [0.0] * 17
            self.control_publisher_.publish(movement_msg)
            self.state = "IDLE"

    def start_pose_detected(self, keypoints):
        # Define the logic for detecting the start pose
        return keypoints.get("RIGHT_WRIST", {}).get("y", float('inf')) < keypoints.get("MOUTH_RIGHT", {}).get("y", float('inf'))

    def stop_pose_detected(self, keypoints):
        # Define the logic for detecting the stop pose
        return keypoints.get("RIGHT_WRIST", {}).get("y", 1) < keypoints.get("MOUTH_RIGHT", {}).get("y", 1)


def main(args=None):
    rclpy.init(args=args)
    gesture_recognition_node = GestureRecognitionNode()
    try:
        rclpy.spin(gesture_recognition_node)
    except KeyboardInterrupt:
        gesture_recognition_node.get_logger().info("Shutting down Gesture Recognition Node")
    finally:
        gesture_recognition_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
