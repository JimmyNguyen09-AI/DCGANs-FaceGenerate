import torch.nn as nn
import torch
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
class Generator(nn.Module):
    def __init__(self,nz= 100,ngf= 64,num_channels = 3):
        super(Generator,self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz,ngf*8,4,1,0,bias  = False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf , num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    def forward(self,x):
        return self.model(x)
class Discriminator(nn.Module):
    def __init__(self,nc = 3,ndf = 64):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            #layer1
            nn.Conv2d(nc,ndf,4,2,1,bias = False),
            nn.LeakyReLU(0.2,inplace = True),
            #layer2
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            #layer3
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # layer4
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # layer5
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self,input):
        return self.model(input)
if __name__ == '__main__':
    input = torch.randn(64,100,1,1)
    model = Generator()
    model2 = Discriminator()
    output = model(input)
    print(output.shape)
    print(model2(output).shape)

