import torch
import torchvision.transforms as transforms
from PIL import Image

def print_examples(model,device,dataset):
    transform=transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    model.eval()
    test_img1=transform(Image.open("test/dog.jpg").convert("RGB")).unsqueeze(0)
    print("example 1:")
    print("output: "+ " ".join(model.caption_image(test_img1.to(device),dataset.vocab)))


    test_img2=transform(Image.open("test/man.jpg").convert("RGB")).unsqueeze(0)
    print("example 2:")
    print("output: "+ " ".join(model.caption_image(test_img2.to(device),dataset.vocab)))
    
    test_img3=transform(Image.open("test/duck.jpg").convert("RGB")).unsqueeze(0)
    print("example 3:")
    print("output: "+ " ".join(model.caption_image(test_img3.to(device),dataset.vocab)))
    
    test_img4=transform(Image.open("test/kids.jpg").convert("RGB")).unsqueeze(0)
    print("example 4:")
    print("output: "+ " ".join(model.caption_image(test_img4.to(device),dataset.vocab)))
    
    test_img5=transform(Image.open("test/war.jpg").convert("RGB")).unsqueeze(0)
    print("example 5:")
    print("output: "+ " ".join(model.caption_image(test_img5.to(device),dataset.vocab)))
    


    model.train()

def save_checkpoints(state,filename="best.pt"):
    print("saving checkpoint")
    torch.save(state,filename)

def load_checkpoints(checkpoint,model,optimier):
    print("loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimier.load_state_dict(checkpoint['optimizer'])
    step=checkpoint['step']
    return step
