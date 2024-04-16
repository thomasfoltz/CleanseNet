import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import Generator, Discriminator
from pertubation import fgsm_attack, PurturbedDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    purturbed_trainset = PurturbedDataset(trainset, fgsm_attack, device)
    dataloader = torch.utils.data.DataLoader(purturbed_trainset, batch_size=32, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters())
    optimizer_D = torch.optim.Adam(discriminator.parameters())

    epochs = 10
    for epoch in range(epochs):
        for i, (purturbed_imgs, real_imgs, _) in enumerate(dataloader):
            print(f"Epoch: {epoch}, Batch: {i}")
            real_images = real_imgs.to(device)
            purturbed_images = purturbed_imgs.to(device)
            b_size = real_images.size(0)

            # Create real and fake labels
            real_labels = torch.ones(b_size, 1).to(device)
            fake_labels = torch.zeros(b_size, 1).to(device)

            # Train the Discriminator
            optimizer_D.zero_grad()
            # print(real_images.size())
            outputs = discriminator(real_images)
            # print(outputs.size(), real_labels.size())
            real_loss = criterion(outputs, real_labels)
            real_score = outputs

            # Generate fake images
            fake_images = generator(purturbed_images)
            # print(fake_images.size())
            outputs = discriminator(fake_images.detach())
            # print(outputs.size(), fake_labels.size())
            fake_loss = criterion(outputs, fake_labels)
            fake_score = outputs

            # Backprop and optimize
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train the Generator
            optimizer_G.zero_grad()
            fake_images = generator(purturbed_images)
            outputs = discriminator(fake_images)

            # Measure generator's ability to fool the discriminator
            g_loss = criterion(outputs, real_labels)

            # Backprop and optimize
            g_loss.backward()
            optimizer_G.step()


    torch.save(generator.state_dict(), 'generator_model.pth')
    torch.save(discriminator.state_dict(), 'discriminator_model.pth')

    # torchvision.utils.save_image(trainset[1][0], 'image_original.png')
    # torchvision.utils.save_image(purturbed_trainset[1][0], 'image_purturbed.png')
