import einops
import torch
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import Resize,ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import MultiheadAttention
from einops.layers.torch import Rearrange
from einops import repeat



to_tensor = [Resize((144,144)),ToTensor()]

class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self,image,target):
        for t in self.transforms:
            image = t(image)
        return image,target


def show_images(images,num_samples=20,cols=4):
    plt.figure(figsize=(15,15))
    idx = int(len(dataset) / num_samples)
    print(images)

    for i,img in enumerate(images):
        if i % idx == 0:
            plt.subplot(int(num_samples/cols) + 1,cols,int(i/idx) + 1)
            plt.imshow(img[0])


dataset = OxfordIIITPet(root=".",download=True,transforms=Compose(to_tensor))
show_images(dataset)


# Patching

class PatchEmbeddings(nn.Module):
    def __init__(self,in_channels=3,patch_size = 8,embed_size=128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1 = patch_size,p2 = patch_size),
            nn.Linear(patch_size * patch_size * in_channels,embed_size)
        )

    def forward(self,x):
        x = self.projection(x)
        return x


# -- quick test --

sample_datapoints = torch.unsqueeze(dataset[0][0],0)
print('Initial Shape :' , sample_datapoints.shape)

embedding = PatchEmbeddings()(sample_datapoints)
print('Patches shape :-' , embedding.shape)



# -------------- Attention -------------

class Attention(nn.Module):
    def __init__(self,dim,n_heads,dropout):
        super().__init()
        self.n_heads = n_heads
        self.att = MultiheadAttention(dim,n_heads,dropout)

        self.q = torch.nn.Linear(dim,dim)
        self.k = torch.nn.Linear(dim,dim)
        self.v = torch.nn.Linear(dim,dim)

    def forward(self,x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn_output,attn_output_weights = self.att(q,k,v)
        return attn_output




# ----- Pre Normalization ----
class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self,x):
        return self.fn(self.norm(x))


norm = PreNorm(128,Attention(dim=128,n_heads=4,dropout=0.))
norm(torch.ones((1,5,128))).shape


# ----------- Linear Layer ----------

class FeedForward(nn.Sequential):
    def __init__(self,dim,hidden_dim,dropout = 0.):
        super().__init__(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )

ff = FeedForward(dim=128,hidden_dim=256)
ff(torch.ones((1,5,128))).shape


# Residual error to amplify the information process

class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn

    def forward(self,x):
        res = x
        x = self.fn(x)
        x += res
        return x


residual_att = Residual(Attention(dim=128,n_heads=4,dropout=0.))
residual_att(torch.ones((1,5,128))).shape


# ------------------------ Vision Transformers ----------------------

class ViT(nn.Module):
    def __init__(self,ch=3,img_size=144,patch_size=8,emb_dim=32,n_layers=4,out_dim=37,dropout=0.1,heads=2):
        super(ViT,self).__init__()

        # Attributes
        self.channels = ch
        self.heights = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        # Patching
        self.patch_embedding = PatchEmbeddings(in_channels=ch,patch_size=patch_size,embed_size=emb_dim)


        # Learnable Params
        num_patches = (img_size // patch_size) ** 2
        self.pos_embeddings = nn.Parameter(
            torch.randn(1,num_patches + 1,emb_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_dim))


        # Transformer encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                Residual(PreNorm(emb_dim,Attention(emb_dim,n_heads = heads,dropout=dropout))),
                Residual(PreNorm(emb_dim,FeedForward(emb_dim,emb_dim,dropout=dropout)))
            )
            self.layers.append(transformer_block)
            

        # Classification Head
        self.head = nn.Sequential(nn.LayerNorm(emb_dim),nn.Linear(emb_dim,out_dim))

    def forward(self,img):
        # Get Patch embedding vectors
        x= self.patch_embedding(img)
        b,n,_ = x.shape

        # Add cls token to inputs
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d',b=b)
        x = torch.cat((cls_tokens,x),dim=1)
        x += self.pos_embeddings[:,:(n+1)]

        # Transformers layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # Output based on classification token
        return self.head(x[:,0,:])



model = ViT()
model(torch.ones((1,3,144,144)))






