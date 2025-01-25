import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

CLASSES = ['Normal', 'Lung_Opacity', 'Viral Pneumonia', 'COVID']
COUNT = 100 # short database size per class, modify at will
import os
from PIL import Image
from torch.utils.data import Dataset


class COVID19Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Data initialization
        self.root_dir = root_dir
        self.transform = transform
        self.classes = CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)} # Gives classes a label
        self.image_paths = [] # (image_path, label)
        
        for cls_name in self.classes:
            count = 0
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                # if count >= COUNT: # -----> uncomment for short database 
                #     break
                if img_name.endswith(".png"): # only png files 
                    img_path = os.path.join(cls_folder, img_name)
                    self.image_paths.append((img_path, self.class_to_idx[cls_name]))
                count += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #Returns tuple
        img_path, label = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB') 
        except Exception as e:
            print(f"Cant load image at: {img_path} - error: {e}")
            return None, None
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def display_batch(self, indexes):
        num_images = len(indexes)
        cols = int(num_images**0.5)  
        rows = (num_images + cols - 1) // cols  

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        axes = axes.flatten()  

        for i, idx in enumerate(indexes):
            if idx >= len(self):
                print(f"Index {idx} out of bounds")
                continue
            image, label = self[idx] 
            if image is None: 
                print(f"Cant load image with index: {idx}")
                continue # abandons unloaded image
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy() # (C, H, W) --> (H, W, C)
            axes[i].imshow(image)
            axes[i].set_title(self.classes[label])
            axes[i].axis('off')
        for ax in axes[num_images:]:
            ax.axis('off')
        plt.tight_layout()
        plt.show()


