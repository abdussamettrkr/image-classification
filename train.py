from math import degrees
import torch
import torch.optim.Adam
from torch.utils.tensorboard import SummaryWriter
from dataset.ImageDataset import ImageDataset


import  models.vit as vits

import os, argparse




parser = argparse.ArgumentParser(description="VIT Image Classification Training")
#parser.add_argument('--dataset', default="cifar10", choices=['cifar10','custom'],)
parser.add_argument("--dataset-path", required=True, type=str)
parser.add_argument("--img_size", default=[224,224],type=list)
parser.add_argument("--batch-size",required=True,type=int)
parser.add_argument("--work-path",default = os.path.dirname(os.path.abspath(__file__)),type=str)
parser.add_argument('--model',required=True,choices=['vit_small','vit_tiny','vit_base'])
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
parser.add_argument("--save-freq", default=5, type=int)

args = parser.parse_args()



#Prepare logging
experiment_dir_path = os.path.join(args.work_path,"experiments",args.exp_name)
if not os.path.exists(experiment_dir_path):
    os.mkdir(experiment_dir_path)


tb = SummaryWriter(experiment_dir_path)



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    transform = transform.Compose([
        transform.RandomResizedCrop(args.img_size[0],scale=(0.9,1.1)),
        transform.RandomRotation(degrees=(-20,20))
    ])

    train_dataset = ImageDataset(root_dir=args.dataset_path,transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,             
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    model = get_model()
    

def get_model()



if __name__ == '__main__':
    main()
