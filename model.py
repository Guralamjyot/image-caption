import torch
from torch import nn
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class EncoderCnn(nn.Module):
    def __init__(self, embed_size,train_cnn=False):
        super().__init__()
        self.train_CNN=train_cnn
        self.inception=models.inception_v3(weights='DEFAULT')
        self.inception.fc=nn.Linear(self.inception.fc.in_features,embed_size)
        self.relu=nn.ReLU()
        self.times=[]
        self.dropout=nn.Dropout(0.5)


    def forward(self,images):
        #features,_ =self.inception(images)
        features =self.inception(images)
        
        return self.dropout(self.relu(features))
    
class DecoderRnn(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers)
        self.linear=nn.Linear(embed_size,vocab_size)
        self.dropout=nn.Dropout(0.5)
    
    def forward(self,features,captions):
        embeddins=self.dropout(self.embed(captions))
        embeddins=torch.cat((features.unsqueeze(0),embeddins),dim=0)
        hiddens,_ =self.lstm(embeddins)
        output=self.linear(hiddens)
        return output
    

class Attention(nn.Module):
    def __init__(self,encoder_dim,decoder_dim,attention_dim):
        super().__init__()
        self.encoder_att= nn.Linear(encoder_dim,attention_dim)
        self.decoder_att= nn.Linear(decoder_dim,attention_dim)
        self.full_att= nn.Linear(attention_dim,1)
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)

    def forward(self,encoder_out,decoder_hidden):
        att1=self.encoder_att(encoder_out)
        att2=self.decoder_att(decoder_hidden)
        att=self.full_att(self.relu(att1+att2.unsqueeze(1))).squeeze(2)
        alpha=self.softmax(att)
        attention_weighted_encoding=(encoder_out*alpha.unsqueeze(2)).sum(dim=1)
        
        return attention_weighted_encoding,alpha


class Attentiondeocder(nn.Module):
    def __init__(self,attention_dim,embed_size,decoder_dim,vocab_size,encoder_dim) :
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_size = embed_size
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention=Attention(encoder_dim,decoder_dim,attention_dim)

        self.embed=nn.Embedding(vocab_size,embed_size)
        self.decode_step=nn.LSTMCell(embed_size+encoder_dim,decoder_dim,bias=True)
        self.init_h=nn.Linear(encoder_dim,decoder_dim)
        self.init_c=nn.Linear(encoder_dim,decoder_dim)
        self.f_beta=nn.Linear(encoder_dim,decoder_dim)
        self.sigmoid=nn.Sigmoid()
        self.fc=nn.Linear(decoder_dim,vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c


    def forward(self,encoder_out,encoded_caption,caption_len):
        batch_size=encoder_out.size(0)
        encoder_dim=encoder_out.size(-1)
        vocab_size=self.vocab_size

        encoder_out=encoder_out.view(batch_size,-1,encoder_dim)
        num_pixels=encoder_out.size(1)

        caption_len,sort_ind=caption_len.squeeze(1).sort(dim=0,descending=True)
        encoder_out=encoder_out[sort_ind]
        encoded_caption=encoded_caption[sort_ind]

        embeddings= self.embed(encoded_caption)

        h, c = self.init_hidden_state(encoder_out)

        decode_len= (caption_len-1).tolist()

        predictions=torch.zeros(batch_size,max(decode_len),vocab_size).to(device)
        alphas=torch.zeros(batch_size,max(decode_len),vocab_size).to(device)

        for t in range(max(decode_len)):
            batch_size_t = sum([l>t for l in decode_len])
            attention_weighted_encoding,alpha= self.attention(encoder_out[:batch_size_t],h[:batch_size_t])
            gate=self.sigmoid(self.f_beta(h[:batch_size_t]))
            h,c=self.decode_step(
                torch.cat([embeddings[:batch_size_t,t,:],attention_weighted_encoding],dim=1),
                (h[:batch_size_t],c[:batch_size_t]))
            preds=self.fc(h)
            predictions[:batch_size_t,t,:]=preds
            alphas[:batch_size_t,t,:]=alphas

        return predictions,encoded_caption,decode_len,alphas,sort_ind

        






class CnntoRnn(nn.Module):
    def __init__(self, embed_size,hidden_size,vocab_size,num_layers):
        super().__init__()
        self.encoder=EncoderCnn(embed_size)
        self.decoder=DecoderRnn(embed_size,hidden_size,vocab_size,num_layers)

    def forward(self,images,captions):
        features=self.encoder(images)
        outputs=self.decoder(features,captions)
        return outputs 

    def caption_image(self,image,vocabulary,max_length=50):
        result_caption=[]

        with torch.inference_mode():
            x=self.encoder(image).unsqueeze(0)
            states=None

            for _ in range(max_length):
                hiddens,states=self.decoder.lstm(x,states)
                outputs=self.decoder.linear(hiddens.squeeze(0))
                predicted=outputs.argmax(1)
                result_caption.append(predicted.item())
                x= self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()]=="<end>":
                    break
        
        return [vocabulary.itos[idx] for idx in result_caption]