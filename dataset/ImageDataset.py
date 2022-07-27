import os
from PIL import Image

from torch.utils.data import Dataset
from torch.nn.functional import one_hot



class ImageDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.class_dirs = os.listdir(root_dir)
        self.label_dict = self.create_label_dict()

        self.images = self.get_image_paths()


    def create_labels(self):
        label_dict = {}

        for idx,dir_name in enumerate(self.class_dirs):
            label_dict[dir_name] = idx


        return label_dict

    def get_image_paths(self):
        images = []
        for class_dir in self.class_dirs:
            class_path = os.path.join(self.root_dir,class_dir)
            class_label = self.label_dict[class_path]

            for image in os.listdir(class_path):
                image_path = os.path.join(class_path,image)
                
                images.append(image_path,class_label)

        return images




    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        
        img,label = self.images[index]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        label = one_hot(label)
        return img,label