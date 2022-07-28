from logging import root
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

from dataset.ImageDataset import ImageDataset


import  models.vit.vit as vits
import os, argparse, yaml, time
import utils
from tqdm import tqdm




parser = argparse.ArgumentParser(description="VIT Image Classification Training")
#parser.add_argument('--dataset', default="cifar10", choices=['cifar10','custom'],)
parser.add_argument("--dataset-path", required=True, type=str)
parser.add_argument("--val-path", required=True, type=str)
parser.add_argument("--img-size", default=224,type=int)
parser.add_argument("--batch-size",required=True,type=int)
parser.add_argument("--num-workers",default=4,type=int)
parser.add_argument("--epochs",default=300,type=int)
parser.add_argument("--exp-name",required=True,type=str)
parser.add_argument("--config-file",required=True,type=str)
parser.add_argument("--work-path",default = os.path.dirname(os.path.abspath(__file__)),type=str)

parser.add_argument('--model',required=True,choices=['vit_small','vit_tiny','vit_base'])
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")


#Vit
parser.add_argument("--patch-size",default=16,type=int,help="Vision transformer patch size")

parser.add_argument("--save-freq", default=5, type=int)

args = parser.parse_args()



#Prepare logging
experiment_dir_path = os.path.join(args.work_path,"experiments",args.exp_name)
if not os.path.exists(experiment_dir_path):
    os.mkdir(experiment_dir_path)


tb = SummaryWriter(experiment_dir_path)


def train(train_loader,model, criterion, optimizer, epoch,lr_scheduler,model_config,fp16_scaler =None):
    model.train()    
    end = time.time()

    loss_current=0
    for it, (images,targets) in enumerate(pbar :=tqdm(train_loader)):
        images = torch.stack([img.cuda() for img in images])
        targets = targets.cuda().long()
        
        data_time = time.time() - end

        if it >= 100:
            break

        it = len(train_loader)*epoch + it
        

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_scheduler[it]            


        with torch.cuda.amp.autocast(fp16_scaler is not None):
            model_time = time.time()                        
            output = model(images)
            model_time = time.time() - model_time
            loss = criterion(output, targets)
        
        optimizer.zero_grad()

        if fp16_scaler is None:
            loss.backward()
            if model_config['clip_grad']:
                param_norms = utils.clip_gradients(model, model_config.clip_grad)            

            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if model_config['clip_grad']:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(model, model_config['clip_grad'])
            
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            

        batch_size = targets.size(0)
        soft = nn.Softmax(dim=1)
        output = soft(output)

        total_correct = (torch.argmax(output)==torch.argmax(targets)).float().sum()
        accuracy = total_correct/batch_size


        
        tb.add_scalar('loss',loss.item(),it)
        tb.add_scalar('lr',lr_scheduler[it],it)
        tb.add_scalar('data_loadtime',data_time,it)
        tb.add_scalar('model_time',model_time,it)

        end = time.time() 

        loss_current += loss.item()

        pbar.set_postfix({'lr':lr_scheduler[it],'loss':loss.item(),'model':model_time,'data':data_time})
        
    loss_current /= len(train_loader)    


    return loss_current

def evaluate(model,data_loader,criterion):
    model.eval()

    total_correct = 0
    batch_size=0
    total_loss = 0
    current_iter = 0
    current_acc = 0
    for it, (images,targets) in enumerate(pbar :=tqdm(data_loader)):
        images = torch.stack([img.cuda() for img in images])
        targets = targets.cuda().long()
        output = model(images)
        loss = criterion(output, targets)

        batch_size = targets.size(0)
        soft = nn.Softmax(dim=1)
        output = soft(output)

        correct = (torch.argmax(output)==torch.argmax(targets)).float().sum()
        total_correct += correct

        current_iter+=1
        current_acc = total_correct/(current_iter*batch_size)

        total_loss += loss.item()
        pbar.set_postfix({'Loss':loss.item(),'Accuracy':current_acc})


    return total_correct/(batch_size*current_iter), total_loss/current_iter



def main():

    with open(args.config_file) as stream:
        model_config = yaml.safe_load(stream)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    transform_train = T.Compose([
        
        T.RandomResizedCrop(args.img_size,scale=(0.9,1)),
        T.RandomRotation(degrees=(-20,20)),
        T.Resize(args.img_size),    
        T.ToTensor()
    ])

    transform_val = T.Compose([
        T.Resize(args.img_size),            
        T.ToTensor()
    ])

    train_dataset = ImageDataset(root_dir=args.dataset_path,transform=transform_train)
    val_dataset = ImageDataset(root_dir=args.val_path,transform=transform_val)



    train_loader = torch.utils.data.DataLoader(
        train_dataset,             
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True
    )    

    val_loader = torch.utils.data.DataLoader(
        val_dataset,             
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True
    )    

    criterion = nn.CrossEntropyLoss()

    model = get_model(device,train_dataset.num_classes)

    print("Number of parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = torch.optim.AdamW(model.parameters())

    lr_schedule = utils.cosine_scheduler(
        model_config['lr'] * args.batch_size / 256., 
        model_config['min_lr'],
        model_config['epochs'], len(train_loader),
        warmup_epochs=model_config['warmup_epochs'],
    )

    fp16_scaler = None
    if model_config['use_fp16']:        
        fp16_scaler = torch.cuda.amp.GradScaler()

    last_epoch = -1

    for epoch in range(last_epoch + 1, args.epochs):
        loss = train(train_loader,model,criterion,optimizer,epoch,lr_schedule,model_config,fp16_scaler)

        val_acc,val_loss = evaluate(model,val_loader,criterion)

        print('Epoch {}, val accuracy: {}'.format(epoch,val_acc))
        
        if val_loss < best_loss:
            best_loss = val_loss
        
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'last_epoch': epoch,            
            'best_loss': val_loss
        }
        
        if epoch % args.save_freq == 0 and epoch != 0:
            state_path = os.path.join(experiment_dir_path,str(epoch))
            utils.save_checkpoint(state_dict,state_path)

        state_path = os.path.join(experiment_dir_path,"last")
        utils.save_checkpoint(state_dict,state_path)

        if loss == best_loss:
            state_path = os.path.join(experiment_dir_path,"best")
            utils.save_checkpoint(state_dict,state_path)

    
    

def get_model(device,num_classes):
    model = None
    if args.model in vits.__dict__.keys():        
        model = vits.__dict__[args.model](args.patch_size,num_classes=num_classes,img_size=args.img_size)        
    model.to(device)
    return model



if __name__ == '__main__':
    main()
