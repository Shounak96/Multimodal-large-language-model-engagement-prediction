# stuff you might have to install
# !pip install transformers torch torchvision pillow accelerate datasets scikit-learn

import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import (
    BertTokenizer, BertModel,
    ViTImageProcessor, ViTModel,
    AutoImageProcessor, AutoModel
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# load dataset
train_json_path = "../data/train_with_eng.json"  # TODO: Update this path
test_json_path = "../data/test_with_eng.json"   # TODO: Update this path
val_json_path = "../data/validation_with_eng.json"     # TODO: Update this path

with open(train_json_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open(test_json_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
    
with open(val_json_path, 'r', encoding='utf-8') as f:
    val_data = json.load(f)

print(f"Total samples: {len(train_data) + len(test_data) + len(val_data)}")

# label mapping
label_to_id = {
    "Very Low": 0,
    "Low": 1,
    "Average": 2,
    "High": 3,
    "Very High": 4
}
id_to_label = {v: k for k, v in label_to_id.items()}


valid_train_data = []
for sample in train_data:
    entry = {}
    if os.path.exists(sample['image']):
        entry = {key: value for key, value in sample.items() if key not in ['engagement_score', 'engagement']}
        valid_train_data.append(entry)
    else:
        print(f"Warning: Image not found: {sample['image']}")

print(f"Valid samples: {len(valid_train_data)}")

valid_test_data = []
for sample in test_data:
    if os.path.exists(sample['image']):
        entry = {key: value for key, value in sample.items() if key not in ['engagement_score', 'engagement']}
        valid_test_data.append(entry)
    else:
        print(f"Warning: Image not found: {sample['image']}")

print(f"Valid samples: {len(valid_test_data)}")

valid_val_data = []
for sample in val_data:
    if os.path.exists(sample['image']):
        entry = {key: value for key, value in sample.items() if key not in ['engagement_score', 'engagement']}
        valid_val_data.append(entry)
    else:
        print(f"Warning: Image not found: {sample['image']}")
print(f"Valid samples: {len(valid_val_data)}")

train_data = valid_train_data
test_data = valid_test_data
val_data = valid_val_data


# Split dataset into train/val/test
# train_data, temp_data = train_test_split(valid_data, test_size=0.3, random_state=42, 
#                                           stratify=[s['engagement_label'] for s in valid_data])
# val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42,
#                                         stratify=[s['engagement_label'] for s in temp_data])

# print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# BERTfor text
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# ViT for images
vit_model_name = "google/vit-base-patch16-224"
image_processor = ViTImageProcessor.from_pretrained(vit_model_name)

print(f"BERT Model: {bert_model_name}")
print(f"ViT Model: {vit_model_name}")

class MultimodalEngagementDataset(Dataset):
    def __init__(self, data, tokenizer, image_processor, label_to_id, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.label_to_id = label_to_id
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # combine title and desc
        text = f"{sample['title']} [SEP] {sample['tag']} [SEP] {sample['description']}"
        text_inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        image = Image.open(sample['image']).convert('RGB')
        image_inputs = self.image_processor(images=image, return_tensors='pt')
        
        label = self.label_to_id[sample['engagement_label']]
        
        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': image_inputs['pixel_values'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

class MultimodalEngagementClassifier(nn.Module):
    def __init__(self, bert_model_name, vit_model_name, num_classes=5, 
                 fusion_method='concat', freeze_bert=False, freeze_vit=False):
        """
            fusion_method: 'concat', 'attention', or 'gated'
            freeze_bert: Whether to freeze BERT weights
            freeze_vit: Whether to freeze ViT weights
        """
        super().__init__()
        
        # Text encoder (BERT)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size  # 768 FOR BASAE
        
        # Image encoder (ViT)
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vit_hidden_size = self.vit.config.hidden_size  # 768 FOR BASE
        
        # Freeze params for efficiency
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        self.fusion_method = fusion_method
        
        # Fusion layers based on method
        if fusion_method == 'concat':
            # Simple concatenation
            fusion_dim = self.bert_hidden_size + self.vit_hidden_size
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        
        elif fusion_method == 'attention':
            # Cross-attention fusion
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.bert_hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            # Project ViT features to match BERT dimension
            self.vit_projection = nn.Linear(self.vit_hidden_size, self.bert_hidden_size)
            
            self.classifier = nn.Sequential(
                nn.Linear(self.bert_hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        
        elif fusion_method == 'gated':
            # Gated fusion mechanism
            self.text_gate = nn.Sequential(
                nn.Linear(self.bert_hidden_size, self.bert_hidden_size),
                nn.Sigmoid()
            )
            self.image_gate = nn.Sequential(
                nn.Linear(self.vit_hidden_size, self.vit_hidden_size),
                nn.Sigmoid()
            )
            
            fusion_dim = self.bert_hidden_size + self.vit_hidden_size
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, input_ids, attention_mask, pixel_values):
        # Get BERT text features
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output  # [bs, 768]
        
        # Get ViT image features
        image_outputs = self.vit(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output  # [bs, 768]
        
        # Fusion
        if self.fusion_method == 'concat':
            fused_features = torch.cat([text_features, image_features], dim=1)
        
        elif self.fusion_method == 'attention':
            image_projected = self.vit_projection(image_features).unsqueeze(1)
            text_expanded = text_features.unsqueeze(1)
            
            attended_features, _ = self.cross_attention(
                text_expanded, image_projected, image_projected
            )
            fused_features = attended_features.squeeze(1)
        
        elif self.fusion_method == 'gated':
            text_gate = self.text_gate(text_features)
            image_gate = self.image_gate(image_features)
            
            gated_text = text_features * text_gate
            gated_image = image_features * image_gate
            
            fused_features = torch.cat([gated_text, gated_image], dim=1)
        
        logits = self.classifier(fused_features)
        return logits
    
# Fusion method either -  'concat', 'attention', or 'gated'
fusion_method = 'concat'

model = MultimodalEngagementClassifier(
    bert_model_name=bert_model_name,
    vit_model_name=vit_model_name,
    num_classes=5,
    fusion_method=fusion_method,
    freeze_bert=False,  
    freeze_vit=False    
).to(device)

print(f"\nFusion Method: {fusion_method}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


batch_size = 64  #TODO: increase to improve training speed

train_dataset = MultimodalEngagementDataset(train_data, tokenizer, image_processor, label_to_id)
val_dataset = MultimodalEngagementDataset(val_data, tokenizer, image_processor, label_to_id)
test_dataset = MultimodalEngagementDataset(test_data, tokenizer, image_processor, label_to_id)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

# TODO: Incease epochs
num_epochs = 10
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = num_training_steps // 10

from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=2e-3,
    total_steps=num_training_steps,
    pct_start=0.1,
    anneal_strategy='cos'
)

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask, pixel_values)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': loss.item(), 
            'acc': 100 * correct / total,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    return total_loss / len(dataloader), 100 * correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask, pixel_values)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    return total_loss / len(dataloader), accuracy * 100, all_preds, all_labels


best_val_acc = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 70)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
    
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_multimodal_engagement_model.pth')
        print(f"Saved best model with val acc: {val_acc:.2f}%")

torch.save({
    'model_state_dict': model.state_dict(),
    'label_to_id': label_to_id,
    'id_to_label': id_to_label,
    'history': history,
    'fusion_method': fusion_method,
    'bert_model_name': bert_model_name,
    'vit_model_name': vit_model_name
}, 'multimodal_engagement_classifier_complete.pth')