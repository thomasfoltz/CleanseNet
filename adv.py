import torch
from torch.utils.data import Dataset
import foolbox as fb
import numpy as np

class AdvEMNIST(Dataset):
    def __init__(self, dataset, model, eps=0.03):
        self.dataset = dataset
        self.model = model
        self.eps = eps
        self.fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = image.unsqueeze(0)  # Add batch dimension
        # Convert to numpy as Foolbox works with numpy arrays
        image_np = image.numpy()
        label_np = np.array([label])

        # Apply FGSM attack
        attack = fb.attacks.FGSM()
        raw, clipped, is_adv = attack(self.fmodel, image_np, criterion=label_np, epsilons=self.eps)

        # Convert back to PyTorch tensor and remove batch dimension
        adv_image = torch.from_numpy(clipped[0]).squeeze(0)

        return adv_image, label