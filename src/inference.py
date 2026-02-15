import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from typing import Dict, Union

from config import DEVICE, IMAGE_SIZE, CLASSES
from model import get_model

class DefectPredictor:
    def __init__(self, model_path: str):
        self.device = DEVICE
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = get_model(num_classes=len(CLASSES))
        
        # Load weights (map_location ensures it works on CPU even if trained on GPU)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle cases where checkpoint saves 'state_dict' or the model directly
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path: str) -> Dict[str, Union[str, float]]:
        """
        Predicts the defect class for a single image.
        """
        if not os.path.exists(image_path):
            return {"error": "Image not found"}

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0) # Add batch dimension (1, C, H, W)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            return {
                "filename": os.path.basename(image_path),
                "prediction": CLASSES[predicted_idx.item()],
                "confidence": round(confidence.item(), 4)
            }

if __name__ == "__main__":
    # Example usage for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference script for Steel Defect Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, default="models/model_best.pth", help="Path to .pth model")
    
    args = parser.parse_args()
    
    predictor = DefectPredictor(args.model)
    result = predictor.predict(args.image)
    print(result)