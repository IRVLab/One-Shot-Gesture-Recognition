import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from keypoints_interfaces.msg import HandKeypoints, HandKeypointsArray
from std_msgs.msg import Bool
import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX models
def load_onnx_model(weight_path):
    """
    Loads an ONNX model with optimized settings.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 4

    session = ort.InferenceSession(
        weight_path,
        options,
        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    return session

# Resize image for YOLOv7 and add padding
def letterbox(img, new_shape=(416, 416)):
    """
    Resizes the image while maintaining the aspect ratio and pads the rest.
    """
    shape = img.shape[:2]  # Current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # Padding

    dw /= 2  # Divide padding into 2 sides
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, ratio, (dw, dh)

# Crop and process image for classification
def preprocess_for_classifier(img, bbox, img_size=(192, 192)):
    """
    Crops and resizes detected hands for the gesture classifier.
    """
    x1, y1, x2, y2 = bbox
    hand_img = img[y1:y2, x1:x2]  # Crop hand
    hand_img = cv2.resize(hand_img, img_size)  # Resize for model
    hand_img = hand_img.astype(np.float32) / 255.0  # Normalize
    hand_img = np.transpose(hand_img, (2, 0, 1))  # Convert to CHW format
    hand_img = np.expand_dims(hand_img, 0)  # Add batch dimension
    return hand_img

# Extract hand keypoints from heatmap
def get_max_preds(heatmaps):
    """
    Extracts keypoints from the heatmap predictions of the classifier.
    """
    num_keypoints = heatmaps.shape[1]
    preds = np.zeros((num_keypoints, 2), dtype=np.float32)

    for i in range(num_keypoints):
        heatmap = heatmaps[0, i, :, :]
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        preds[i] = [x, y]

    return preds

# Check if detected keypoints form a Vulcan Salute
def is_vulcan_salute(landmarks):
    """
    Determines if the detected hand gesture is a Vulcan Salute.
    """
    if landmarks is None or len(landmarks) < 21:
        return False

    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    distance_index_middle = np.linalg.norm(index_tip - middle_tip)
    distance_middle_ring = np.linalg.norm(middle_tip - ring_tip)
    distance_ring_pinky = np.linalg.norm(ring_tip - pinky_tip)

    return 1.2 * distance_index_middle < distance_middle_ring and 1.2 * distance_ring_pinky < distance_middle_ring

# Detect hands and keypoints using YOLOv7 and classifier
def detect_hand_and_gesture(frame, hand_detector, classifier):
    """
    Runs hand detection using YOLOv7 mini and classifies gestures.
    """
    # Preprocess frame for YOLO (416x416 for detection)
    img, ratio, dwdh = letterbox(frame, new_shape=(416, 416))
    img = img[:, :, ::-1]  # Convert BGR to RGB
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0).astype(np.float32) / 255.0

    # Run YOLO model (hand detection)
    input_name = hand_detector.get_inputs()[0].name
    detections = hand_detector.run(None, {input_name: img})[0]

    hands_detected = []
    keypoints_list = []
    bbox_list = []
    vulcan_salute_detected = False

    for det in detections:
        # confidence, x0, y0, x1, y1 = det[:5]  # Unpack bounding box
        _, x0, y0, x1, y1, _, score = det
        # print(f"det = {det}")
        if score > 0.5:  # Confidence threshold
            box = np.array([x0, y0, x1, y1]) - np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(int).tolist()
            hands_detected.append(box)

    for bbox in hands_detected:
        # Preprocess for classification (192x192 for classifier)
        hand_img = preprocess_for_classifier(frame, bbox, img_size=(192, 192))

        # Run classifier model
        input_name_cls = classifier.get_inputs()[0].name
        label_pred, heatmap_pred = classifier.run(None, {input_name_cls: hand_img})

        # Extract keypoints
        keypoints = get_max_preds(heatmap_pred)

        # Check if Vulcan Salute detected
        if is_vulcan_salute(keypoints):
            vulcan_salute_detected = True

        keypoints_list.append(keypoints)
        bbox_list.append(bbox)

    return keypoints_list, bbox_list, vulcan_salute_detected

class HandDetectionNode(Node):
    def __init__(self):
        super().__init__('hand_detection_node')
        self.bridge = CvBridge()
        self.frame_count = 0

        # Load YOLO and classifier models
        yolov7_mini_path = "/home/irvlab/rishi_ws/src/robo_chat_gest/model_data/yolov7-tiny-diver.onnx"
        classifier_path = "/home/irvlab/rishi_ws/src/robo_chat_gest/model_data/gesture-classifier.onnx"
        self.hand_detector = load_onnx_model(yolov7_mini_path)
        self.classifier = load_onnx_model(classifier_path)
        
        self.subscription = self.create_subscription(
            Image,
            '/zed/zed_node/left/image_rect_color',
            self.image_callback,
            10
        )

        self.keypoints_publisher = self.create_publisher(HandKeypointsArray, 'hand_keypoints', 10)
        self.vulcan_salute_publisher = self.create_publisher(Bool, 'vulcan_salute_detected', 10)

        self.get_logger().info("Hand Detection Node started")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        keypoints_list, bbox_list, vulcan_salute_detected = detect_hand_and_gesture(frame, self.hand_detector, self.classifier)

        if keypoints_list:
            keypoints_array = HandKeypointsArray()
            
            for keypoints in keypoints_list:
                for idx, (x, y) in enumerate(keypoints):
                    keypoint = HandKeypoints()
                    keypoint.name = f"Point_{idx}"
                    keypoint.x = float(x)
                    keypoint.y = float(y)
                    keypoint.z = float(0)
                    keypoints_array.keypoints.append(keypoint)
            
            self.keypoints_publisher.publish(keypoints_array)

        # Publish Vulcan Salute detection
        self.vulcan_salute_publisher.publish(Bool(data=bool(vulcan_salute_detected)))

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
