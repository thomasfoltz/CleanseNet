import argparse
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import Generator, Discriminator
from perturbation import fgsm_attack, PurturbedDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training script for CleanseNet')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to train GAN')
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    perturbed_testset = PurturbedDataset(trainset, fgsm_attack, device)
    dataloader = torch.utils.data.DataLoader(perturbed_testset, batch_size=64, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    l1_loss = nn.L1Loss()
    optimizer_G = torch.optim.Adam(generator.parameters())
    optimizer_D = torch.optim.Adam(discriminator.parameters())

    for epoch in range(args.epochs):
        for i, (perturbed_imgs, real_imgs, _) in enumerate(dataloader):
            print(f"Epoch: {epoch}, Batch: {i}")
            real_images = real_imgs.to(device)
            perturbed_images = perturbed_imgs.to(device)
            b_size = real_images.size(0)

            real_labels = torch.ones(b_size, 1).to(device)
            fake_labels = torch.zeros(b_size, 1).to(device)

            verified_labels = discriminator(real_images)
            generated_images = generator(perturbed_images)
            generated_labels = discriminator(generated_images)

            real_loss = criterion(verified_labels, real_labels)
            fake_loss = criterion(generated_labels, fake_labels)
            total_d_loss = real_loss + fake_loss

            pixel_loss = l1_loss(generated_images, real_images)
            total_g_loss = pixel_loss

            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            total_d_loss.backward(retain_graph=True)
            total_g_loss.backward(retain_graph=True)
            optimizer_D.step()
            optimizer_G.step()

            print(f"total_g_loss: {total_g_loss}")
            print(f"total_d_loss: {total_d_loss}")

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
