import mediapipe as mp
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


class FaceCrop:
    def __init__(self, images_path:Path, save_path:Path,scale=1):
        self.images_path = images_path
        self.save_path = save_path
        self.scale = scale
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=0.1)

    def cropAll(self):
        for image in tqdm(list(self.images_path.rglob('*.jpg'))):
            self.crop(image)
    def crop(self, image_path:Path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(image)
        saveImagePath = self.save_path / image_path.parent.parent.name / image_path.name
        saveImagePath.parent.mkdir(parents=True, exist_ok=True)  
        if results.detections:
            for detection in results.detections:
                x, y = int(detection.location_data.relative_bounding_box.xmin * image.shape[1]), int(
                    detection.location_data.relative_bounding_box.ymin * image.shape[0])
                w, h = int(detection.location_data.relative_bounding_box.width * image.shape[1]), int(
                    detection.location_data.relative_bounding_box.height * image.shape[0])
                #apply scale 
                x = int(x - (w * self.scale - w) / 2)
                y = int(y - (h * self.scale - h) / 2)
                w = int(w * self.scale)
                h = int(h * self.scale)

                crop_image = image[y:y + h, x:x + w]
                if crop_image.shape[0] == 0 or crop_image.shape[1] == 0:
                    print(f'No face detected in {image_path}')
                    return
                crop_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)
                crop_image = cv2.resize(crop_image, (160, 160))
                cv2.imwrite(str(saveImagePath), crop_image)
        else:
            print(f'No face detected in {image_path}')

if __name__ == '__main__':
    #get images path from command line
    image_path = Path("/home/st392/fsl_groups/grp_nlp/compute/RFW/data/")
    save_path = Path("/home/st392/fsl_groups/grp_nlp/compute/RFW/CroppedImages/")
    save_path.mkdir(parents=True, exist_ok=True)
    faceCrop = FaceCrop(image_path, save_path)
    faceCrop.cropAll()
