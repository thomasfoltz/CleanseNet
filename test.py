import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from perturbation import fgsm_attack, PurturbedDataset
from models import load_model, Generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test script for CleanseNet')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    perturbed_testset = PurturbedDataset(testset, fgsm_attack, device)

    generator = load_model('generator.pth', Generator(), device)

    dataloader = torch.utils.data.DataLoader(perturbed_testset, batch_size=32)
    for images, _, _ in dataloader:
        images = images.to(device)
        results = generator(images)

        torchvision.utils.save_image(images[3], 'test_original_perturbed.png')
        torchvision.utils.save_image(results[3], 'test_cleaned_perturbed.png')
        break
