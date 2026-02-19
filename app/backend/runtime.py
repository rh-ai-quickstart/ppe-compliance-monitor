import numpy as np
import cv2
import os

# import matplotlib.pyplot as plt
from ovmsclient import make_grpc_client, make_http_client
from pydantic import BaseModel


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]
    scale: float


class Runtime:
    def __init__(self):
        self.service_url = os.getenv("SERVICE_URL")
        self.input_name = os.getenv("MODEL_INPUT_NAME")
        self.model_name = os.getenv("MODEL_NAME")
        self.model_version = int(os.getenv("MODEL_VERSION"))
        # Choose the inference function based on the environment
        openshift_mode = os.getenv("OPENSHIFT", "false").lower() == "true"
        self.inference_fun = (
            self.remote_inference if openshift_mode else self.local_inference
        )
        # 10 PPE classes from the model metadata
        self.CLASSES = {
            0: "Hardhat",
            1: "Mask",
            2: "NO-Hardhat",
            3: "NO-Mask",
            4: "NO-Safety Vest",
            5: "Person",
            6: "Safety Cone",
            7: "Safety Vest",
            8: "machinery",
            9: "vehicle",
        }

    def preprocess_image(self, image: np.ndarray):
        """
        Preprocess the image for the model.
        """
        # Read the input image
        # image = cv2.imread("team.jpg")
        [height, width, _] = image.shape

        # Prepare a square image for inference (letterbox padding)
        length = max(height, width)
        padded = np.zeros((length, length, 3), np.uint8)
        padded[0:height, 0:width] = image
        image = padded
        scale = length / 640

        # Preprocess: normalize, resize to 640x640, swap BGR->RGB, produce NCHW blob
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1 / 255, size=(640, 640), swapRB=True
        )
        return blob, scale

    def postprocess_image(
        self, outputs: np.ndarray, input_image: np.ndarray, scale: float
    ) -> list[Detection]:
        """
        Postprocess the tensor for the model.
        """
        # Transpose to (1, 8400, 14) for easier per-detection iteration
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(
                classes_scores
            )
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply Non-Maximum Suppression
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.20, 0.45, 0.5)

        # colors = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

        # def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        #     label = f"{self.CLASSES[class_id]} ({confidence:.2f})"
        #     color = colors[class_id]
        #     cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        #     cv2.putText(
        #         img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        #     )

        detections: list[Detection] = []
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = Detection(
                class_id=class_ids[index],
                class_name=self.CLASSES[class_ids[index]],
                confidence=scores[index],
                bbox=box,
                scale=scale,
            )
            detections.append(detection)
            # draw_bounding_box(
            #     input_image,
            #     class_ids[index],
            #     scores[index],
            #     round(box[0] * scale),
            #     round(box[1] * scale),
            #     round((box[0] + box[2]) * scale),
            #     round((box[1] + box[3]) * scale),
            # )

        # # Save the output image
        # cv2.imwrite("output.jpeg", input_image)

        # # Display inline (convert BGR -> RGB for matplotlib)
        # plt.figure(figsize=(14, 10))
        # plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.title(f"YOLO PPE Detection via OVMS — {len(detections)} detections")
        # plt.show()

        # # Print detections summary
        # for d in detections:
        #     print(f"{d['class_name']}: {d['confidence']:.2f}")

        return detections

    def inference(self, image: np.ndarray) -> np.ndarray:
        """
        Inference the image for the model.
        """
        return self.inference_fun(image)

    def local_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Local inference the image for the model.
        """
        client = make_grpc_client(self.service_url)
        inputs = {self.input_name: image}
        response = client.predict(inputs, self.model_name, self.model_version)
        return response

    def remote_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Remote inference the image for the model.
        """
        http_url = self.service_url.replace("https://", "").replace("http://", "")

        client = make_http_client(http_url)

        inputs = {self.input_name: image}
        return client.predict(inputs, self.model_name, self.model_version, timeout=60.0)

    def run(self, image: np.ndarray) -> list[Detection]:
        """
        Run the inference for the image.
        """
        blob, scale = self.preprocess_image(image)
        outputs = self.inference(blob)
        return self.postprocess_image(outputs, image, scale)
