import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from facenet_pytorch import MTCNN
from src.models.vit_model import ViTDeepfakeClassifier

class DeepfakePipeline:
    def __init__(self, model_weights_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = MTCNN(margin=40, keep_all=False, post_process=False, device=self.device)
        self.model = ViTDeepfakeClassifier().to(self.device)

        state = torch.load(model_weights_path, map_location=self.device)
        self.model.load_state_dict(state if "model_state_dict" not in state else state["model_state_dict"])
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def analyze_video(self, video_path: str, num_frames: int = 10):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: return {"error": "Invalid video"}

        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        scores = []
        suspicious_images = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            face_tensor = self.detector(img_pil)

            if face_tensor is not None:
                face_img = T.ToPILImage()(face_tensor)
                input_tensor = self.transform(face_img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logit = self.model(input_tensor)
                    prob = torch.sigmoid(logit).item()
                    scores.append(prob)

                    # If high probability of being fake, save the frame for the UI
                    if prob > 0.70:
                        # We store a tuple: (Image, Caption)
                        suspicious_images.append((face_img, f"Frame {idx}: {prob:.2f} Score"))

        cap.release()
        if not scores: return {"error": "No faces detected"}

        mean_score = float(np.mean(scores))
        prediction = "FAKE" if mean_score > 0.5 else "REAL"

        return {
            "prediction": prediction,
            "confidence": mean_score if prediction == "FAKE" else 1 - mean_score,
            "gallery": suspicious_images
        }
