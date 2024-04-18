import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from perturbation import fgsm_attack, PurturbedDataset
from models import load_model, Generator, Discriminator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test script for CleanseNet')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    perturbed_testset = PurturbedDataset(testset, fgsm_attack, device)

    generator = load_model('generator.pth', Generator(), device)
    discriminator = load_model('discriminator.pth', Discriminator(), device)

    dataloader = torch.utils.data.DataLoader(perturbed_testset, batch_size=32)
    for perturbed, original, original_labels in dataloader:
        perturbed = perturbed.to(device)
        original = original.to(device)
        cleaned = generator(perturbed)

        predetermined_labels = discriminator(perturbed)
        determined_labels = discriminator(cleaned)

        predetermined_labels = torch.transpose(predetermined_labels, 0, 1)
        determined_labels = torch.transpose(determined_labels, 0, 1)

        print(original_labels)
        print(predetermined_labels)
        print(determined_labels)

        print(torch.median(predetermined_labels), torch.median(determined_labels))

        torchvision.utils.save_image(original[3], 'test_original.png')
        torchvision.utils.save_image(perturbed[3], 'test_perturbed.png')
        torchvision.utils.save_image(cleaned[3], 'test_cleaned.png')
        break
