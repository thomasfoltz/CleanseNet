import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pertubation import fgsm_attack, PurturbedDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    purturbed_trainset = PurturbedDataset(trainset, fgsm_attack, device)

    torchvision.utils.save_image(trainset[1][0], 'image_original.png')
    torchvision.utils.save_image(purturbed_trainset[1][0], 'image_purturbed.png')
