import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import ResNet18

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("当前使用设备：",device)

batch_size=128
model_path='best_model78.pth'

test_data=torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    transform=torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
    ])
)

test_loader=DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
)

def evaluate(model,test_loader,criterion,device):
    model.eval()

    running_loss=0.0
    correct=0
    total=0

    with torch.no_grad():
        for images,labals in test_loader:
            images=images.to(device)
            labals=labals.to(device)

            outputs=model(images)

            loss=criterion(outputs,labals)
            running_loss+=loss.item()

            _,predicted=outputs.max(1)
            total+=labals.size(0)
            correct+=predicted.eq(labals).sum().item()
        
    test_loss=running_loss/len(test_loader)
    test_acc=100*correct/total
    return test_loss,test_acc

if __name__=='__main__':
    model=ResNet18().to(device)

    state_dict=torch.load(model_path,map_location=device)
    model.load_state_dict(state_dict)

    criterion=nn.CrossEntropyLoss()

    test_loss,test_acc=evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.2f}%")

            