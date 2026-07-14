import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import ResNet18

classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)

def load_image(image_path):
    transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])
    image=Image.open(image_path).convert("RGB")
    image=transform(image)
    image=image.unsqueeze(0)

    return image

def predict(image_path,model_path,device):
    model=ResNet18().to(device)

    state_dict=torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    image=load_image(image_path)
    image=image.to(device)

    with torch.no_grad():
        outputs=model(image)
        prob=F.softmax(outputs,dim=1)
        
        top_probs,top_idx=torch.topk(prob,k=3,dim=1)

    top3_results=[]

    for prob_value,classes_idx in zip(top_probs[0],top_idx[0]):
        class_name=classes[classes_idx.item()]
        confidence=prob_value.item()*100
        top3_results.append((class_name,confidence))
    return top3_results

def show_image_with_predict(image_path,top3_results):
    image=Image.open(image_path).convert('RGB')

    title = "Top-3 Predictions\n"

    for rank,(class_name,confidence) in enumerate(top3_results,start=1):
        title+=f"{rank}.{class_name}:{confidence:.2f}%\n"
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',type=str,required=True)
    parser.add_argument('--model',type=str,default='best_model50.pth')
    args=parser.parse_args()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    top3_results=predict(
        image_path=args.image,
        model_path=args.model,
        device=device
    )
    print('Top3--Predictions')
    for rank,(class_name,confidence) in enumerate(top3_results):
        print(f"{rank}.{class_name}:{confidence:.2f}%\n")
    
    show_image_with_predict(args.image,top3_results)