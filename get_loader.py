import os
from typing import Any  
import pandas as pd 
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

spacy_eng=spacy.load('en_core_web_sm')

class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos={0:'<pad>',1:'<eos>',2:'<start>',3:'<unk>'}
        self.stoi={'<pad>':0,'<eos>':1,'<start>':2,'<unk>':3}
        self.freq_threshold=freq_threshold

    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self,sentence_list):
        frequencies={}
        idx=4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word]=1

                else:
                    frequencies[word]+=1

                if frequencies[word]==self.freq_threshold:
                    self.stoi[word]=idx
                    self.itos[idx]=word
                    idx+=1

    def numericalize(self,text):
        tokenized_text=self.tokenizer_eng(text)

        return[ 
            self.stoi[token] if token in self.stoi else self.stoi['<unk>'] for token in tokenized_text
        ]
    

class flikerdataset(Dataset):
    def __init__(self,root,captions_file,transform=None,freq_threshold=5):
        super().__init__()
        self.root=root
        self.df=pd.read_csv(captions_file)
        self.transform=transform

        self.imgs=self.df['image']
        self.captions=self.df['caption']

        self.vocab=Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        caption=self.captions[index]
        img_id=self.imgs[index]
        img=Image.open(os.path.join(self.root,img_id)).convert('RGB')

        if self.transform is not None:
            img=self.transform(img)

        numericalized_caption=[self.vocab.stoi['<start>']]
        numericalized_caption+=self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi['<eos>'])

        return img,torch.tensor(numericalized_caption)
    

class MyCollate:
    def __init__(self,pad_idx) :
        self.pad_idx=pad_idx

    def __call__(self,batch) :
        imgs=[item[0].unsqueeze(0) for item in batch]
        imgs=torch.cat(imgs,dim=0)
        targets=[item[1] for item in batch]
        targets=pad_sequence(targets,batch_first=False,padding_value=self.pad_idx)

        return imgs,targets
    

def get_loader(
        root,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
):
    dataset=flikerdataset(root,annotation_file,transform=transform)
    pad_idx=dataset.vocab.stoi['<pad>']

    loader=DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
        
    return loader,dataset


if __name__=="__main__":
    transform=transforms.Compose(
        [transforms.Resize((224,224)),transforms.ToTensor(),]
    )

    loader, dataset=get_loader('Images/','captions.txt',transform=transform)

    for idx, (imgs,captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)