import csv, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

SEED=42
EPOCHS=int(os.getenv('EPOCHS','6'))
BATCH=int(os.getenv('BATCH_SIZE','256'))
EXPERIMENT=os.getenv('EXPERIMENT','adam_aug')
OUT=Path('assignment_output')
DATA=Path('data')


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class CustomReLU(nn.Module):
    def forward(self,x): return torch.clamp(x,min=0.0)

class MeanImageSubtract:
    def __init__(self,mean): self.mean=mean
    def __call__(self,x): return x-self.mean

class CNN(nn.Module):
    def __init__(self):
        super().__init__(); R=CustomReLU
        self.net=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1,bias=False),nn.BatchNorm2d(32),R(),
            nn.Conv2d(32,32,3,padding=1,bias=False),nn.BatchNorm2d(32),R(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1,bias=False),nn.BatchNorm2d(64),R(),
            nn.Conv2d(64,64,3,padding=1,bias=False),nn.BatchNorm2d(64),R(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1,bias=False),nn.BatchNorm2d(128),R(),nn.MaxPool2d(2),
            nn.Flatten(),nn.Linear(128*4*4,256,bias=False),nn.BatchNorm1d(256),R(),nn.Linear(256,10))
    def forward(self,x): return self.net(x)

@torch.no_grad()
def mean_image():
    ds=datasets.CIFAR10(DATA,train=True,download=True,transform=transforms.ToTensor())
    dl=DataLoader(ds,batch_size=512,shuffle=False,num_workers=2)
    s=torch.zeros(3,32,32); n=0
    for x,_ in dl: s+=x.sum(0); n+=x.size(0)
    m=s/n; OUT.mkdir(exist_ok=True); torch.save(m,OUT/'mean_image.pt')
    plt.figure(figsize=(4,4)); plt.imshow(np.clip(m.permute(1,2,0).numpy(),0,1)); plt.axis('off'); plt.title('CIFAR-10 Mean Image'); plt.tight_layout(); plt.savefig(OUT/'mean_image.png',dpi=180); plt.close()
    print('mean channels',m.mean((1,2)).tolist(),flush=True); return m

def loaders(mean,aug):
    tr=[]
    if aug: tr += [transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(0.5)]
    tr += [transforms.ToTensor(),MeanImageSubtract(mean)]
    te=[transforms.ToTensor(),MeanImageSubtract(mean)]
    train=datasets.CIFAR10(DATA,train=True,download=False,transform=transforms.Compose(tr))
    test=datasets.CIFAR10(DATA,train=False,download=True,transform=transforms.Compose(te))
    return DataLoader(train,batch_size=BATCH,shuffle=True,num_workers=2),DataLoader(test,batch_size=BATCH,shuffle=False,num_workers=2)

def make_optimizer(name,model):
    if name=='Adam': return torch.optim.Adam(model.parameters(),lr=1e-3,betas=(0.9,0.999))
    if name=='SGD+Momentum': return torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
    if name=='AdamW': return torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(0.9,0.999),weight_decay=0.01)
    raise ValueError(name)

def epoch_pass(model,dl,criterion,opt,device):
    train=opt is not None; model.train(train); total=correct=0; loss_sum=0.0
    context=torch.enable_grad() if train else torch.no_grad()
    with context:
        for x,y in dl:
            x=x.to(device); y=y.to(device)
            if train: opt.zero_grad(set_to_none=True)
            logits=model(x); loss=criterion(logits,y)
            if train: loss.backward(); opt.step()
            loss_sum+=loss.item()*y.size(0); correct+=(logits.argmax(1)==y).sum().item(); total+=y.size(0)
    return loss_sum/total,100.0*correct/total

def main():
    specs={'adam_no_aug':('Adam',False),'adam_aug':('Adam',True),'sgd_momentum_aug':('SGD+Momentum',True),'adamw_aug':('AdamW',True)}
    if EXPERIMENT not in specs: raise ValueError(EXPERIMENT)
    seed_all(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',device,'experiment',EXPERIMENT,flush=True)
    mean=mean_image(); opt_name,aug=specs[EXPERIMENT]; train,test=loaders(mean,aug)
    seed_all(); model=CNN().to(device); criterion=nn.CrossEntropyLoss(); opt=make_optimizer(opt_name,model)
    writer=SummaryWriter(str(OUT/'runs'/EXPERIMENT)); history=[]; best=-1.; best_epoch=0; t0=time.time()
    for ep in range(1,EPOCHS+1):
        tl,ta=epoch_pass(model,train,criterion,opt,device); vl,va=epoch_pass(model,test,criterion,None,device)
        writer.add_scalar('Loss/train',tl,ep); writer.add_scalar('Loss/test',vl,ep); writer.add_scalar('Accuracy/train',ta,ep); writer.add_scalar('Accuracy/test',va,ep)
        history.append({'epoch':ep,'train_loss':tl,'train_acc':ta,'test_loss':vl,'test_acc':va})
        if va>best: best=va; best_epoch=ep
        print(f'{EXPERIMENT} epoch {ep}/{EPOCHS} train_loss={tl:.4f} train_acc={ta:.2f}% test_loss={vl:.4f} test_acc={va:.2f}%',flush=True)
    writer.close(); d=OUT/'results'/EXPERIMENT; d.mkdir(parents=True,exist_ok=True)
    with open(d/'history.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    summary={'run':EXPERIMENT,'optimizer':opt_name,'augmentation':aug,'epochs':EPOCHS,'best_test_acc':best,'best_epoch':best_epoch,'final_train_loss':history[-1]['train_loss'],'final_train_acc':history[-1]['train_acc'],'final_test_loss':history[-1]['test_loss'],'final_test_acc':history[-1]['test_acc'],'elapsed_seconds':time.time()-t0}
    (d/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
