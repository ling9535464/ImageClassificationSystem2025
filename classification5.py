import os
import hashlib
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import torch.cuda.amp as amp
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, all_data_dir, abnormal_data_dir, transform=None, abnormal_transform=None, cache_hashes=True):
        self.all_data_dir = all_data_dir
        self.abnormal_data_dir = abnormal_data_dir
        self.transform = transform  #正常样本的增强
        self.abnormal_transform = abnormal_transform  #异常样本的额外增强
        self.img_hashes = []
        self.labels = []
        self.hash_to_path = {}
        self.label_map = {'normal': 0, 'abnormal': 1}
        self.cache_hashes = cache_hashes
        self.hashes_file = "hashes.csv"
        self._load_data()

    def _load_data(self):
        if self.cache_hashes and os.path.exists(self.hashes_file):
            self._load_from_cache()
        else:
            self._compute_and_save_hashes()

        for root, dirs, files in os.walk(self.all_data_dir):
            for file in files:
                if file.endswith('.jpg'):
                    img_path = os.path.join(root, file)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        img_hash = self._get_image_hash(img_path)
                        label = self.label_map['abnormal'] if img_hash in self.abnormal_img_hashes else self.label_map['normal']
                        self.labels.append(label)
                        self.img_hashes.append(img_hash)
                        self.hash_to_path[img_hash] = img_path
                    except (IOError, SyntaxError) as e:
                        print(f"Error loading {img_path}: {e}")
                        pass

    def _compute_and_save_hashes(self):
        self.abnormal_img_hashes = set()
        for root, dirs, files in os.walk(self.abnormal_data_dir):
            for file in files:
                if file.endswith('.jpg'):
                    img_path = os.path.join(root, file)
                    img_hash = self._get_image_hash(img_path)
                    self.abnormal_img_hashes.add(img_hash)
        if self.cache_hashes:
            self._save_to_cache()

    def _load_from_cache(self):
        self.abnormal_img_hashes = set()
        with open(self.hashes_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                self.abnormal_img_hashes.add(row[0])

    def _save_to_cache(self):
        with open(self.hashes_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            for img_hash in self.abnormal_img_hashes:
                writer.writerow([img_hash])

    def _get_image_hash(self, img_path):
        with open(img_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def __getitem__(self, index):
        img_hash = self.img_hashes[index]
        img_path = self.hash_to_path[img_hash]
        label = self.labels[index]
        img = Image.open(img_path).convert('RGB')
        
        if label == 1 and self.abnormal_transform:
            img = self.abnormal_transform(img)
        elif self.transform:
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return len(self.img_hashes)


normal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

abnormal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class RecallLoss(nn.Module):
    def __init__(self, beta=2.0, class_weights=None):
        super().__init__()
        self.beta = beta
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights)
        probabilities = torch.softmax(inputs, dim=1)
        predicted_positive = (probabilities[:, 1] > 0.15) 
        tp = (predicted_positive & (targets == 1)).float().sum()
        fn = ((~predicted_positive) & (targets == 1)).float().sum()
        recall = tp / (tp + fn + 1e-6)

        recall_penalty = self.beta * (1 - recall)  
        return ce_loss + recall_penalty


class PretrainedResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = True
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class EarlyStopping:
    def __init__(self, patience=10, delta=0.005):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.best_recall = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, recall):
        if self.best_loss is None or self.best_recall is None:
            self.best_loss = val_loss
            self.best_recall = recall
        elif val_loss > self.best_loss - self.delta: #or recall < self.best_recall - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_recall = recall
            self.counter = 0

def main():
    all_data_dir = 'C:/Users/17628/Desktop/learning/AAAAproject/2021-2023all'  #固定路径，后续需要修改使得可以选择路径
    abnormal_data_dir = 'C:/Users/17628/Desktop/learning/AAAAproject/2021-2023abnormal'
    
    dataset = CustomDataset(
        all_data_dir, abnormal_data_dir,
        transform=normal_transform,
        abnormal_transform=abnormal_transform,
        cache_hashes=True
    )
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_indices = list(range(len(dataset)))
    all_labels = dataset.labels
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir='runs/improved_experiment')

    for fold, (train_indices, val_test_indices) in enumerate(kfold.split(all_indices, all_labels)):
        print(f"Fold {fold + 1}/5")
        val_indices, test_indices = train_test_split(
            val_test_indices, test_size=0.5,
            stratify=[all_labels[i] for i in val_test_indices],
            random_state=42
        )

        train_data = Subset(dataset, train_indices)
        val_data = Subset(dataset, val_indices)
        test_data = Subset(dataset, test_indices)

        sample_weights = [15 if train_data.dataset.labels[idx] == 1 else 1 for idx in train_indices]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        num_workers = min(4, multiprocessing.cpu_count() // 2)
        train_loader = DataLoader(train_data, batch_size=64, sampler=sampler, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=num_workers)

        model = PretrainedResNet(num_classes=2).to(device)
        class_weights = torch.tensor([1.0, 15.0]).to(device)
        criterion = RecallLoss(beta=2.0, class_weights=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
        early_stopping = EarlyStopping(patience=10, delta=0.001)
        scaler = amp.GradScaler()

        for epoch in range(50):
            #训练阶段
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch + 1}/50, Loss: {epoch_loss:.4f}')
            writer.add_scalar(f'Fold_{fold+1}/Loss/train', epoch_loss, epoch)

            #验证阶段
            model.eval()
            val_loss = 0.0
            correct = 0
            y_true, y_pred = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted = (probabilities[:, 1] > 0.15).long()
                    correct += (predicted == labels).sum().item()
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
            val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = correct / len(val_loader.dataset)
            val_report = classification_report(y_true, y_pred, target_names=['normal', 'abnormal'], output_dict=True)
            val_recall_abnormal = val_report['abnormal']['recall']
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Abnormal Recall: {val_recall_abnormal:.4f}')
            print("Confusion Matrix:")
            print(confusion_matrix(y_true, y_pred))
            
            writer.add_scalar(f'Fold_{fold + 1}/Loss/val', val_loss, epoch)
            writer.add_scalar(f'Fold_{fold + 1}/Accuracy/val', val_accuracy, epoch)
            writer.add_scalar(f'Fold_{fold + 1}/Recall/abnormal', val_recall_abnormal, epoch)

            scheduler.step(val_loss)

            early_stopping(val_loss, val_recall_abnormal)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break


        #测试阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        test_loss = test_loss / len(test_loader.dataset)
        test_accuracy = correct / len(test_loader.dataset)

        test_report = classification_report(y_true, y_pred, target_names=['normal', 'abnormal'], output_dict=True)

        precision_abnormal = test_report['abnormal']['precision']
        recall_abnormal = test_report['abnormal']['recall']
        f1_score_abnormal = test_report['abnormal']['f1-score']

        if precision_abnormal + recall_abnormal > 0:
            f2_score_abnormal = 5 * (precision_abnormal * recall_abnormal) / (4 * precision_abnormal + recall_abnormal)
        else:
            f2_score_abnormal = 0

        print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Abnormal Recall: {recall_abnormal:.4f}, F2-Score: {f2_score_abnormal:.4f}')

        print("Test Classification Report:")
        print(test_report)

        writer.add_scalars(f'Fold_{fold + 1}/Test/Metrics', {
            'Accuracy': test_accuracy,
            'Precision': precision_abnormal,
            'Recall': recall_abnormal,
            'F1-Score': f1_score_abnormal,
            'F2-Score': f2_score_abnormal
        }, epoch)

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    writer.close()

if __name__ == '__main__':
    main()
