import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import ResNet18

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("当前使用设备：",device)

batch_size=128
learning_rate=0.1
num_epochs=10

train_transform=transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

train_data=torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    transform=train_transform,
    download=True
)
test_data=torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    transform=test_transform,
    download=True
)

train_loader=DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True
)
test_loader=DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False
)

model=ResNet18(num_classes=10).to(device)

criterion=nn.CrossEntropyLoss()

optimizer=optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=5e-4
)

def train_one_epoch(model,train_loader,criterion,optimizer,device):
    model.train()

    running_loss=0.0
    correct=0
    total=0

    for batch_idx,(images,labels) in enumerate(train_loader):
        images=images.to(device)
        labels=labels.to(device)

        outputs=model(images)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        _,predicted=outputs.max(1)
        total+=labels.size(0)

        correct+=predicted.eq(labels).sum().item()

        if (batch_idx+1)%100==0:
            print(
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {running_loss / (batch_idx + 1):.4f} "
                f"Acc: {100.0 * correct / total:.2f}%"
            )
    epoch_loss=running_loss/len(train_loader)
    epoch_acc=100*correct/total
    return epoch_loss,epoch_acc

def evaluate(model,test_loader,criterion,device):
    model.eval()

    ruuning_loss=0.0
    correct=0
    total=0

    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            loss=criterion(outputs,labels)

            ruuning_loss+=loss

            _,predicted=outputs.max(1)
            total+=labels.size(0)
            correct+=predicted.eq(labels).sum().item()
    
    test_loss=ruuning_loss/len(test_loader)
    test_acc=100*correct/total
    return test_loss,test_acc

if __name__ == "__main__":
    best_acc=0.0
    for epoch in range(num_epochs):
        print(f'Epoch[{epoch+1}/{num_epochs}]')

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        test_loss,test_acc=evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc: {test_acc:.2f}%")

        if test_acc>best_acc:
            best_acc=test_acc
            torch.save(model.state_dict(),f'best_model{epoch+1}.pth')
            print(f'save the best model with acc:{best_acc}')
        print('*'*50)