import argparse
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import Discriminator, UNet
from perturbation import fgsm_attack, PurturbedDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training script for CleanseNet')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to train GAN')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    generator = UNet().to(device)
    discriminator = Discriminator().to(device)

    perturbed_testset = PurturbedDataset(trainset, fgsm_attack, device)
    dataloader = torch.utils.data.DataLoader(perturbed_testset, batch_size=64, shuffle=True)

    alpha = 0.1
    adversarial_loss, l1_loss = nn.BCELoss(), nn.L1Loss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        for i, (perturbed_imgs, real_imgs, _) in enumerate(dataloader):
            real_images = real_imgs.to(device)
            perturbed_images = perturbed_imgs.to(device)
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)

            optimizer_G.zero_grad()
            generated_images = generator(perturbed_images)
            generated_labels = discriminator(generated_images)
            generated_adv_loss = adversarial_loss(generated_labels, real_labels)
            generated_l1_loss = l1_loss(generated_images, real_images)
            generator_loss = generated_adv_loss * alpha + generated_l1_loss * (1-alpha)
            generator_loss.backward(retain_graph=True)
            optimizer_G.step()

            optimizer_D.zero_grad()
            verified_labels = discriminator(real_images)
            real_loss = adversarial_loss(verified_labels, real_labels)
            fake_loss = adversarial_loss(generated_labels, fake_labels)
            discriminator_loss = (real_loss + fake_loss)/2
            discriminator_loss.backward()
            optimizer_D.step()

            print(f"Epoch: {epoch}, Batch: {i}, g_loss={generator_loss}, d_loss={discriminator_loss}")

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
