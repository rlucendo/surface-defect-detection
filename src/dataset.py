import os
import glob
from typing import List, Tuple, Optional
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SteelSurfaceDataset(Dataset):
    """
    Custom Dataset for Steel Surface Defects.
    Reads images from a directory structure where subfolders represent classes.
    """

    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None, mode: str = 'train'):
        """
        Args:
            root_dir (str): Path to the root directory containing class subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'valid', or 'test'. Used to filter data if needed.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_map = {}
        
        # Load data structure
        self._load_dataset()

    def _load_dataset(self):
        """Internal method to populate image paths and labels."""
        # Get all subdirectories (classes)
        classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
        if not classes:
            raise FileNotFoundError(f"No classes found in {self.root_dir}. Check directory structure.")

        for idx, class_name in enumerate(classes):
            self.class_map[idx] = class_name
            class_path = os.path.join(self.root_dir, class_name)
            
            # Get all images (jpg, png, bmp)
            image_files = glob.glob(os.path.join(class_path, "*.*"))
            
            for img_path in image_files:
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[any, int]:
        """
        Args:
            idx (int): Index of the item.

        Returns:
            tuple: (image, label) where label is the index of the class.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image and convert to RGB (some might be grayscale)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image or handle error appropriately in production
            image = Image.new('RGB', (224, 224)) 

        if self.transform:
            image = self.transform(image)

        return image, label