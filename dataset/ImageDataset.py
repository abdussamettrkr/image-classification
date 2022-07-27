from cProfile import label
import os
from PIL import Image

from torch.utils.data import Dataset
import numpy as np



class ImageDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.class_dirs = os.listdir(root_dir)
        self.label_dict = self.create_label_dict()
        self.num_classes = len(self.label_dict)

        self.images = self.get_image_paths()



    def create_label_dict(self):
        label_dict = {}

        for idx,dir_name in enumerate(self.class_dirs):
            label_dict[dir_name] = idx


        return label_dict

    def get_image_paths(self):
        images = []
        for class_dir in self.class_dirs:
            class_path = os.path.join(self.root_dir,class_dir)
            class_label = self.label_dict[class_dir]

            for image in os.listdir(class_path):
                image_path = os.path.join(class_path,image)
                
                images.append([image_path,class_label])

        return images




    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        
        image_path,label = self.images[index]
        img = Image.open(image_path)
        


        if self.transform:
            img = self.transform(img)
        
        labels = np.zeros(self.num_classes)
        labels[label] = 1

        return img,label