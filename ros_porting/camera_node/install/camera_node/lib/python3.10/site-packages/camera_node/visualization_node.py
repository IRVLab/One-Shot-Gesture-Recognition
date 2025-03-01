import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from keypoints_interfaces.msg import HandKeypointsArray, PoseKeypointsArray
from cv_bridge import CvBridge
import cv2
from datetime import datetime

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        self.bridge = CvBridge()
        self.frame = None
        self.hand_keypoints = None
        self.pose_keypoints = None

        self.image_pub = self.create_publisher(
            Image, 'processed_frames', 10
        )

        self.image_sub = self.create_subscription(
            Image, '/zed/zed_node/left/image_rect_color', self.image_callback, 10
        )
        self.hand_keypoints_sub = self.create_subscription(
            HandKeypointsArray, 'hand_keypoints', self.hand_keypoints_callback, 10
        )
        self.pose_keypoints_sub = self.create_subscription(
            PoseKeypointsArray, 'pose_keypoints', self.pose_keypoints_callback, 10
        )
        self.out = None
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = f"output_{timestamp}.avi"

        self.get_logger().info("Visualization Node started")

    def image_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.hand_keypoints:
            for kp in self.hand_keypoints.keypoints:
                x, y = int(kp.x * self.frame.shape[1]), int(kp.y * self.frame.shape[0])
                cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
        if self.pose_keypoints:
            for kp in self.pose_keypoints.keypoints:
                x, y = int(kp.x * self.frame.shape[1]), int(kp.y * self.frame.shape[0])
                cv2.circle(self.frame, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("Visualization", self.frame)
        
        processed_msg = self.bridge.cv2_to_imgmsg(self.frame, encoding='bgr8')
        self.image_pub.publish(processed_msg)
        self.get_logger().info("Published processed frame")
        
        """if self.out is None:
            self.out = cv2.VideoWriter(
                self.video_path,
                cv2.VideoWriter_fourcc(*'XVID'),
                self.fps,
                (self.frame.shape[1], self.frame.shape[0])
            )
        self.out.write(self.frame)"""
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Exiting visualization")
            self.destroy_node()
            rclpy.shutdown()

    def hand_keypoints_callback(self, msg):
        self.hand_keypoints = msg

    def pose_keypoints_callback(self, msg):
        self.pose_keypoints = msg

    def destroy_node(self):
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    visualization_node = VisualizationNode()
    try:
        rclpy.spin(visualization_node)
    except KeyboardInterrupt:
        visualization_node.get_logger().info("Shutting down Visualization Node")
    finally:
        visualization_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
