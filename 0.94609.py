import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomErasing, CenterCrop
from PIL import Image
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# ---------------------------
# Configuration
# ---------------------------
MIX_ALPHA = 0.15     # Moderate MixUp alpha for balanced regularization
LABEL_SMOOTH = 0.05  # Balanced label smoothing
LR = 0.001
MAX_LR = 0.0015
BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 15
IMG_SIZE = 224       # 224x224 for faster, more stable training

# ---------------------------
# Label Mapping
# ---------------------------
label_map = {"Istanbul": 0, "Ankara": 1, "Izmir": 2}
reverse_label_map = {v: k for k, v in label_map.items()}

def verify_files(df, root_dir, is_test=False):
    valid_rows = []
    for idx in range(len(df)):
        img_name = df.iloc[idx, 0]
        img_path = os.path.join(root_dir, img_name)
        if not os.path.isfile(img_path):
            continue
        if not is_test:
            label = df.iloc[idx, 1]
            if pd.isna(label):
                continue
        valid_rows.append(idx)
    return df.iloc[valid_rows].reset_index(drop=True)

class StreetViewDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        df = pd.read_csv(csv_file)
        df = verify_files(df, root_dir, is_test=is_test)
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            label = torch.tensor(-1, dtype=torch.long)
        else:
            label_str = self.data.iloc[idx, 1]
            # Ensure label_str is valid
            if label_str not in label_map:
                # If unexpected label, default to "Istanbul"
                label_str = "Istanbul"
            label = torch.tensor(label_map[label_str], dtype=torch.long)

        return image, label

def custom_collate_fn(batch):
    return torch.utils.data._utils.collate.default_collate(batch)

# ---------------------------
# MixUp Functions
# ---------------------------
def mixup_data(x, y, alpha=MIX_ALPHA):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam)*criterion(pred, y_b)

# ---------------------------
# Augmentations
# ---------------------------
train_transform = transforms.Compose([
    RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    RandomErasing(p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3), value='random')
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------------------
# Data Loading
# ---------------------------
train_csv = 'train_data.csv'
test_csv = 'test.csv'
train_dir = 'train/'
test_dir = 'test/'

train_data = pd.read_csv(train_csv)
train_data, val_data = train_test_split(train_data, test_size=0.2, stratify=train_data['city'])
train_data.to_csv('train_split.csv', index=False)
val_data.to_csv('val_split.csv', index=False)

train_dataset = StreetViewDataset('train_split.csv', train_dir, transform=train_transform, is_test=False)
val_dataset = StreetViewDataset('val_split.csv', train_dir, transform=test_transform, is_test=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# Model: EfficientNet-B3
# ---------------------------
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LR, steps_per_epoch=steps_per_epoch, epochs=EPOCHS
)

best_f1 = 0.0
no_improve_epochs = 0

# ---------------------------
# Training Loop with Early Stopping
# ---------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        # MixUp each batch
        mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=MIX_ALPHA)

        optimizer.zero_grad()
        outputs = model(mixed_images)
        loss = mixed_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    # Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Validation F1 Score: {f1}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), 'best_model.pth')
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= PATIENCE:
            print("Early stopping triggered.")
            break

# ---------------------------
# Test-Time Augmentation (TTA)
# Normal, flipped, center-cropped
# ---------------------------
test_dataset = StreetViewDataset(test_csv, test_dir, transform=test_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

center_crop_transform = CenterCrop(IMG_SIZE)
predictions = []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        # Normal
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        # Flipped
        flipped_images = torch.flip(images, dims=[3])
        flipped_outputs = model(flipped_images)
        flipped_probs = torch.nn.functional.softmax(flipped_outputs, dim=1)

        # Center cropped
        c_images = center_crop_transform(images.cpu()).to(device)
        c_outputs = model(c_images)
        c_probs = torch.nn.functional.softmax(c_outputs, dim=1)

        avg_probs = (probs + flipped_probs + c_probs) / 3.0
        preds = torch.argmax(avg_probs, dim=1).cpu().numpy()

        # Validate predictions, fallback if needed
        final_preds = []
        for p in preds:
            if p not in reverse_label_map:
                p = 0
            final_preds.append(p)
        predictions.extend(final_preds)

test_data = pd.read_csv('test.csv')
test_data['city'] = [reverse_label_map[p] for p in predictions]
test_data.to_csv('submission.csv', index=False)

print("Done! MixUp, label smoothing, triple TTA, early stopping, and a stable training setup implemented.")
