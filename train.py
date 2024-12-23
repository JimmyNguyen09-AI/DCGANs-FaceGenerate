import random
from model import Generator,Discriminator
import cv2
import os
import shutil
from GANdataset import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from  tqdm import tqdm
manual_seed = 2004
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.use_deterministic_algorithms(True)
def weights_init(m):   #default weight for GANs
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
def get_args():
    parser = argparse.ArgumentParser("Test Argument")
    parser.add_argument("--batch-size","-b",type = int,default=64,help="Batch size")
    parser.add_argument("--num-epochs","-e",type = int,default=50)
    parser.add_argument("--learning-rate","-l",type = float,default = 0.0002,help ="Learning Rate")
    parser.add_argument("--log-path","-p",type = str,default="tensorboard")
    parser.add_argument("--data-path","-d",type = str,default = "./dataset")
    parser.add_argument("--checkpoint-path", "-c", type=str, default="train_model1")
    parser.add_argument("--pretrained-path", "-t", type=str, default=None)
    parser.add_argument("--noise_vector", "-n", type=int, default=100)
    parser.add_argument("--output_image_file", "-o", type=str, default="output")


    args = parser.parse_args()
    return args
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = data_prepare(args.data_path)
    dataloader = DataLoader(dataset = dataset,batch_size=args.batch_size,shuffle = True,num_workers=4)
    num_iter_per_epoch = len(dataloader)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64,args.noise_vector,1,1,device = device)
    optimizerG = torch.optim.Adam(generator.parameters(),lr = args.learning_rate,betas = (0.5,0.999))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    G_losses = []
    D_losses = []
    writer = SummaryWriter(args.log_path)
    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizerD.load_state_dict(checkpoint["optimizerD"])
        optimizerG.load_state_dict(checkpoint["optimizerG"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.isdir(args.output_image_file):
        os.makedirs(args.output_image_file)
    for epoch in range(start_epoch,args.num_epochs):
        progress_bar = tqdm(dataloader,colour = "cyan")
        losses_real = []
        losses_fake = []
        losses_G = []
        for idx, data in enumerate(progress_bar):
            # update Discriminator maximize log(D(x)) + log(1 - D(G(z)))
            discriminator.zero_grad()
            real_image = data[0].to(device)
            batch_size = real_image.size(0)
            label = torch.full((batch_size,),1,dtype = torch.float,device = device)
            output =  discriminator(real_image).view(-1)
            loss_real = criterion(output,label)
            loss_real.backward()
            D_loss_real = output.mean().item()
            # train with fake batch
            noise = torch.randn(batch_size,args.noise_vector,1,1,device = device)
            fake = generator(noise)
            label.fill_(0)
            output = discriminator(fake.detach()).view(-1)
            loss_fake = criterion(output,label)
            loss_fake.backward()
            D_G_loss = output.mean().item()
            loss_D = loss_real + loss_fake
            losses_real.append(loss_real.item())
            losses_fake.append(loss_fake.item())
            optimizerD.step()
            writer.add_scalar("Decriminator/Loss Real", np.mean(losses_real), epoch+1)
            writer.add_scalar("Decriminator/Loss Fake", np.mean(losses_fake), epoch+1)
            # update Generator   maximize log(D(G(z)))
            generator.zero_grad()
            label.fill_(1)
            output = discriminator(fake).view(-1)
            loss_G = criterion(output,label)
            losses_G.append(loss_G.item())
            loss_G.backward()
            D_G_loss2 = output.mean().item()
            optimizerG.step()
            writer.add_scalar("Generator/Loss", D_G_loss2, epoch+1)
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
            progress_bar.set_description("Epoch: {}/{}. LossD: {:0.4f}.  LossG: {:0.4f}".format(epoch+1,args.num_epochs,loss_D,np.mean(losses_G)))
        with torch.no_grad():
            fake_images = generator(fixed_noise).detach().cpu()
        img_grid = vutils.make_grid(fake_images, padding=2, normalize=True)
        # Show and save generated images
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Generated Images at Epoch {epoch+1}")
        plt.imshow(np.transpose(img_grid, (1, 2, 0)))
        plt.savefig(f"{args.output_image_file}/generated_images_epoch_{epoch+1}.png")
        checkpoint = {
            "epoch": epoch + 1,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "optimizerD":optimizerD.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, "last.pt"))
    writer.close()


if __name__ == '__main__':
    args = get_args()
    train(args)