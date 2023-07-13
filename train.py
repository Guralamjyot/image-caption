import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoints, load_checkpoints, print_examples
from get_loader import get_loader
from model import CnntoRnn

def train():
    transform=transforms.Compose(
        [
            transforms.Resize((356,356)),
            transforms.RandomCrop((299,299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset=get_loader(root="Images/",annotation_file='captions.txt',transform=transform,num_workers=4)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    load_model=True
    save_model=True
    train_CNN=False
    infer_only=True


    embed_size=256
    hidden_size=256
    vocab_size=len(dataset.vocab)
    num_layers=1
    leraning_rate=3e-4
    num_epochs=100


    writer=SummaryWriter("runs/")
    step=0

    model= CnntoRnn(embed_size,hidden_size,vocab_size,num_layers).to(device)
    criterion=nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<pad>'])
    optimizer=optim.Adam(model.parameters(),lr=leraning_rate)

    for name,param in model.encoder.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad=True

        else:
            param.requires_grad=train_CNN

    
    if load_model:
        step=load_checkpoints(torch.load('best.pt'),model,optimizer)
        print('loading model')

    model.train()

    if infer_only is not True:
        for epoch in tqdm(range(num_epochs)):
                    
        
            
            for idx, (imgs,captions) in tqdm(enumerate(train_loader),total=len(train_loader),leave=False):
                imgs=imgs.to(device)
                captions=captions.to(device)
                outputs=model(imgs,captions[:-1])
                loss=criterion(outputs.reshape(-1,outputs.shape[2]),captions.reshape(-1))
                writer.add_scalar("training loss",loss.item(),global_step=step)
                step+=1

                optimizer.zero_grad()
                loss.backward(loss)
                optimizer.step()


            if save_model:
                checkpoint={
                    "state_dict":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "step":step,
                }
                save_checkpoints(checkpoint)

    else:
        print_examples(model,device,dataset)

if __name__=="__main__":
    train()

"""  
print_examples(model,device,dataset)
to print examples have to change forward method of encoder, due to some bug with using inception weights, 
cant disable aux logits so have to change unpack them into an empty var, but during inference the inception
unpacks only a single item and not 2.
"""