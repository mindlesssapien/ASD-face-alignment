import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from facenet_pytorch import MTCNN
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import numpy as np

class AlignedFaceDataset(Dataset):
    """
    Custom PyTorch Dataset that applies MTCNN face alignment on the fly.
    For larger datasets, it is highly recommended to run this alignment 
    once and save the images to disk to speed up training.
    """
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        #initialize MTCNN for landmark detection (extracting eyes, nose, mouth)
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device='cuda' if torch.cuda.is_available() else 'cpu')

    def align_face(self, image):
        """
        Calculates the displacement angle using left and right eye coordinates 
        and rotates the picture for final alignment as described in the paper.
        """
        try:
            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
            if landmarks is not None:
                #0:left eye, 1:right eye
                left_eye = landmarks[0][0]
                right_eye = landmarks[0][1]
                
                #angle for rotation
                dy = right_eye[1] - left_eye[1]
                dx = right_eye[0] - left_eye[0]
                angle = math.degrees(math.atan2(dy, dx))
                
                #rotating image to align eyes horizontally
                aligned_img = image.rotate(angle, resample=Image.BICUBIC)
                
                #croped the face using the bounding box
                box = boxes[0]
                aligned_img = aligned_img.crop((box[0], box[1], box[2], box[3]))
                return aligned_img
        except Exception as e:
            pass 
            
        return image

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.align_face(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DATA_DIR = './dataset' 
train_dataset = AlignedFaceDataset(os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = AlignedFaceDataset(os.path.join(DATA_DIR, 'valid'), transform=transform)
test_dataset = AlignedFaceDataset(os.path.join(DATA_DIR, 'test'), transform=transform)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1) 
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.001)

EPOCHS = 30 
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(train_dataset)
    
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = val_loss / len(val_dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

print("\nInitiating Final Testing Phase")
model.eval()
test_preds, test_probs, test_labels = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels_np = labels.numpy()
        
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(float)
        
        test_probs.extend(probs)
        test_preds.extend(preds)
        test_labels.extend(labels_np)

test_acc = accuracy_score(test_labels, test_preds)
test_auc = roc_auc_score(test_labels, test_probs)
test_prec = precision_score(test_labels, test_preds)
test_rec = recall_score(test_labels, test_preds)

print(f"Test Accuracy:  {test_acc * 100:.2f}%")
print(f"Test AUC:       {test_auc * 100:.2f}%")
print(f"Test Precision: {test_prec * 100:.2f}%")
print(f"Test Recall:    {test_rec * 100:.2f}%")