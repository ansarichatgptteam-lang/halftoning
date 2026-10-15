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
EPOCHS=int(os.getenv('EPOCHS','8'))
BATCH=int(os.getenv('BATCH_SIZE','256'))
OUT=Path('assignment_output')
DATA=Path('data')


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class CustomReLU(nn.Module):
    def forward(self,x):
        return torch.clamp(x,min=0.0)


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


def mean_image():
    ds=datasets.CIFAR10(DATA,train=True,download=True,transform=transforms.ToTensor())
    dl=DataLoader(ds,batch_size=512,shuffle=False,num_workers=2)
    s=torch.zeros(3,32,32); n=0
    for x,_ in dl: s+=x.sum(0); n+=x.size(0)
    m=s/n; OUT.mkdir(exist_ok=True); torch.save(m,OUT/'mean_image.pt')
    plt.figure(figsize=(4,4)); plt.imshow(np.clip(m.permute(1,2,0).numpy(),0,1)); plt.axis('off'); plt.title('CIFAR-10 Mean Image'); plt.tight_layout(); plt.savefig(OUT/'mean_image.png',dpi=180); plt.close()
    print('mean channels',m.mean((1,2)).tolist())
    return m


def loaders(mean,aug):
    tr=[]
    if aug: tr += [transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(0.5)]
    tr += [transforms.ToTensor(),MeanImageSubtract(mean)]
    te=[transforms.ToTensor(),MeanImageSubtract(mean)]
    train=datasets.CIFAR10(DATA,train=True,download=False,transform=transforms.Compose(tr))
    test=datasets.CIFAR10(DATA,train=False,download=True,transform=transforms.Compose(te))
    return (DataLoader(train,batch_size=BATCH,shuffle=True,num_workers=2,pin_memory=torch.cuda.is_available()),
            DataLoader(test,batch_size=BATCH,shuffle=False,num_workers=2,pin_memory=torch.cuda.is_available()))


def optimizer(name,model):
    if name=='Adam': return torch.optim.Adam(model.parameters(),lr=1e-3,betas=(0.9,0.999))
    if name=='SGD+Momentum': return torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
    if name=='AdamW': return torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(0.9,0.999),weight_decay=0.01)
    raise ValueError(name)


def pass_epoch(model,dl,crit,opt,dev):
    train=opt is not None; model.train(train); loss_sum=0.; correct=total=0
    ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x,y in dl:
            x=x.to(dev); y=y.to(dev)
            if train: opt.zero_grad(set_to_none=True)
            z=model(x); loss=crit(z,y)
            if train: loss.backward(); opt.step()
            loss_sum += loss.item()*y.size(0); correct += (z.argmax(1)==y).sum().item(); total += y.size(0)
    return loss_sum/total,100*correct/total


def run(run_name,opt_name,aug,mean,dev):
    seed_all(); tr,te=loaders(mean,aug); model=CNN().to(dev); crit=nn.CrossEntropyLoss(); opt=optimizer(opt_name,model)
    writer=SummaryWriter(str(OUT/'runs'/run_name)); hist=[]; best=-1.; best_ep=0; t0=time.time()
    for ep in range(1,EPOCHS+1):
        tl,ta=pass_epoch(model,tr,crit,opt,dev); vl,va=pass_epoch(model,te,crit,None,dev)
        writer.add_scalar('Loss/train',tl,ep); writer.add_scalar('Loss/test',vl,ep); writer.add_scalar('Accuracy/train',ta,ep); writer.add_scalar('Accuracy/test',va,ep)
        hist.append(dict(epoch=ep,train_loss=tl,train_acc=ta,test_loss=vl,test_acc=va))
        if va>best: best=va; best_ep=ep
        print(run_name,ep,round(tl,4),round(ta,2),round(vl,4),round(va,2),flush=True)
    writer.close(); d=OUT/'results'/run_name; d.mkdir(parents=True,exist_ok=True)
    with open(d/'history.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)
    s=dict(run=run_name,optimizer=opt_name,augmentation=aug,epochs=EPOCHS,best_test_acc=best,best_epoch=best_ep,final_train_loss=hist[-1]['train_loss'],final_train_acc=hist[-1]['train_acc'],final_test_loss=hist[-1]['test_loss'],final_test_acc=hist[-1]['test_acc'],elapsed_seconds=time.time()-t0)
    (d/'summary.json').write_text(json.dumps(s,indent=2)); return hist,s


def plot(histories,key,ylabel,title,fname,only=None):
    plt.figure(figsize=(8,5))
    names=only if only else list(histories)
    for n in names: plt.plot([r['epoch'] for r in histories[n]],[r[key] for r in histories[n]],marker='o',label=n)
    plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'figures'/fname,dpi=220); plt.close()


def main():
    seed_all(); OUT.mkdir(exist_ok=True); (OUT/'figures').mkdir(exist_ok=True); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',dev)
    m=mean_image()
    specs=[('adam_no_aug','Adam',False),('adam_aug','Adam',True),('sgd_momentum_aug','SGD+Momentum',True),('adamw_aug','AdamW',True)]
    H={}; S=[]
    for rn,on,aug in specs: H[rn],s=run(rn,on,aug,m,dev); S.append(s)
    with open(OUT/'summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=S[0].keys()); w.writeheader(); w.writerows(S)
    plot(H,'test_acc','Test Accuracy (%)','CIFAR-10 Test Accuracy','comparison_test_accuracy.png')
    plot(H,'train_loss','Training Loss','CIFAR-10 Training Loss / Convergence','comparison_train_loss.png')
    plot(H,'test_acc','Test Accuracy (%)','Effect of Image Augmentation (Adam)','augmentation_effect.png',['adam_no_aug','adam_aug'])
    plot(H,'test_acc','Test Accuracy (%)','Optimizer Comparison with Augmentation','optimizer_accuracy_comparison.png',['adam_aug','sgd_momentum_aug','adamw_aug'])
    plot(H,'train_loss','Loss','TensorBoard Scalars - Training Loss','tensorboard_train_loss.png')
    plot(H,'train_acc','Accuracy (%)','TensorBoard Scalars - Training Accuracy','tensorboard_train_accuracy.png')
    plot(H,'test_acc','Accuracy (%)','TensorBoard Scalars - Test Accuracy','tensorboard_test_accuracy.png')
    by={s['run']:s for s in S}; diff=by['adam_aug']['best_test_acc']-by['adam_no_aug']['best_test_acc']; rank=sorted(['adam_aug','sgd_momentum_aug','adamw_aug'],key=lambda x:by[x]['best_test_acc'],reverse=True)
    text=(f"Augmentation changed Adam best test accuracy by {diff:+.2f} percentage points.\n"+"\n".join(f"{by[x]['optimizer']}: best {by[x]['best_test_acc']:.2f}% at epoch {by[x]['best_epoch']}" for x in rank)+f"\nBest optimizer in this training budget: {by[rank[0]]['optimizer']}.\n")
    (OUT/'auto_discussion.txt').write_text(text)

if __name__=='__main__': main()
