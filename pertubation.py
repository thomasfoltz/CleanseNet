import torch
from torchvision.models import resnet50
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

def fgsm_attack(image, epsilon, gradient):
    sign_data_grad = gradient.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1) 
    return perturbed_image

class PurturbedDataset(Dataset):
    def __init__(self, dataset, attack, device):
        self.dataset = dataset
        self.attack_function = attack
        self.device = device
        self.loss_function = CrossEntropyLoss()
        self.resnet = resnet50().to(self.device)
        self.resnet.eval()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = image.to(self.device).requires_grad_(True)

        label = torch.tensor(label).to(self.device)

        output = self.resnet(image.unsqueeze(0))
        loss = self.loss_function(output, label.unsqueeze(0))

        self.resnet.zero_grad()
        loss.backward()

        perturbed_image = self.attack_function(image, 0.1, image.grad.data)
        return perturbed_image.detach(), image, label