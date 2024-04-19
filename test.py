import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from perturbation import fgsm_attack, PurturbedDataset
from models import load_model, UNet
from torchvision.models import resnet50

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test script for CleanseNet')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


    generator = load_model('generator.pth', UNet(), device)
    orig_correct, perturbed_correct, cleaned_correct = 0, 0, 0
    orig_total, perturbed_total, cleaned_total = 0, 0, 0
    batch = 0

    perturbed_testset = PurturbedDataset(testset, fgsm_attack, device)
    resnet = resnet50(pretrained=True)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, 10)
    resnet.to(device)
    dataloader = torch.utils.data.DataLoader(perturbed_testset, batch_size=32)

    for perturbed, original, original_labels in dataloader:
        print("Batch", batch)
        batch += 1

        original = original.to(device)
        perturbed = perturbed.to(device)
        cleaned = generator(perturbed)

        orig_outputs = resnet(original)
        perturbed_outputs = resnet(perturbed)
        cleaned_outputs = resnet(cleaned)

        _, orig_predicted = torch.max(orig_outputs.data, 1)
        orig_total += original_labels.size(0)
        orig_correct += (orig_predicted == original_labels).sum().item()

        _, perturbed_predicted = torch.max(perturbed_outputs.data, 1)
        perturbed_total += original_labels.size(0)
        perturbed_correct += (perturbed_predicted == original_labels).sum().item()

        _, cleaned_predicted = torch.max(cleaned_outputs.data, 1)
        cleaned_total += original_labels.size(0)
        cleaned_correct += (cleaned_predicted == original_labels).sum().item()

        # torchvision.utils.save_image(original[1], 'test_original.png')
        # torchvision.utils.save_image(perturbed[1], 'test_perturbed.png')
        # torchvision.utils.save_image(cleaned[1], 'test_cleaned.png')

    print(f"original accuracy: {round((orig_correct/orig_total)*100, 2)}%")
    print(f"perturbed accuracy: {round((perturbed_correct/perturbed_total)*100, 2)}%")
    print(f"cleaned accuracy: {round((cleaned_correct/cleaned_total)*100, 2)}%")
