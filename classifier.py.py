import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
#torch.set_num_threads(4)
import json
from torch import nn, optim
from torchvision import models, transforms
from transformers import ViTForImageClassification
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
#from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
from senet import se_resnet_18
import cv2
import random
random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioDataset(Dataset):
    def __init__(self, folder_path, processed_folder):
        self.audio_paths = []
        self.labels = []
        self.processed_folder = processed_folder

        os.makedirs(processed_folder, exist_ok=True)

        for label, subfolder in enumerate(os.listdir(folder_path)):
            label = 1 if subfolder == "MusicCaps" else 0
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):

                all_audio_files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]
                #breakpoint()
                if label == 1:
                    # MusicCaps 文件夹：保留所有文件
                    sampled_files = all_audio_files
                else:
                    # 其他文件夹：随机采样 800 个文件
                    sampled_files = random.sample(all_audio_files, min(800, len(all_audio_files)))
                
                for audio_file in sampled_files:
                    audio_path = os.path.join(subfolder_path, audio_file)
                    self.labels.append(label)
                    # 生成唯一的保存文件路径
                    unique_filename = f"{label}_{subfolder}_{audio_file.split('.wav')[0]}.npy"
                    save_path = os.path.join(self.processed_folder, unique_filename)
                    self.audio_paths.append(save_path)
                    if os.path.exists(save_path):
                        continue
                    self.process_audio(audio_path, save_path, label)

    def process_audio(self, audio_path, save_path, label):
        audio, sr = librosa.load(audio_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Resize and save the mel spectrogram as a .npy file
        mel_spec_db_resized = cv2.resize(mel_spec_db, (224, 224))
        np.save(save_path, mel_spec_db_resized)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):

        processed_file = self.audio_paths[idx]
        mel_spec_db = torch.tensor(np.load(processed_file)).unsqueeze(0)

        return mel_spec_db, self.labels[idx]

def prepare_datasets(folder_path, processed_folder):
    dataset = AudioDataset(folder_path, processed_folder)
    train_val_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class SENetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SENetClassifier, self).__init__()
        self.senet = se_resnet_18(pretrained=True)
        self.senet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.senet.fc = nn.Linear(self.senet.fc.in_features, num_classes)

    def forward(self, x):
        return self.senet(x)

class VGGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=True)  
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)

class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.features[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

class ViTBinaryClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTBinaryClassifier, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")#, num_labels=num_classes)
        self.vit.config.num_labels = 2
        self.vit.classifier = nn.Linear(self.vit.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.conv(x)

        return self.vit(x).logits

class CNNLSTMClassifier(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=256, num_layers=2, bidirectional=True):
        super(CNNLSTMClassifier, self).__init__()
        
        self.cnn = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.lstm = nn.LSTM(512, lstm_hidden_size, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True)
        
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        batch_size, c, h, w = x.size()
        seq_len =1
        x = x.view(batch_size * seq_len, c, h, w)
        
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)  #(batch_size, seq_len, feature_size)

        lstm_out, _ = self.lstm(x)  # lstm_out.shape: (batch_size, seq_len, lstm_output_size)
        
        last_time_step = lstm_out[:, -1, :]
        
        out = self.fc(last_time_step)
        return out

def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, lr, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    train_val_results = []
    for epoch in range(num_epochs):
        model.train()
        model.to(device)
        train_loss = 0.0
        
        for step, (mel_spec, labels) in enumerate(tqdm(train_loader)):
            mel_spec = mel_spec.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for step, (mel_spec, labels) in enumerate(tqdm(val_loader)):
                mel_spec = mel_spec.to(device)
                labels = labels.to(device)
                outputs = model(mel_spec)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        train_val_results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
    end_time = time.time()  
    total_time = end_time - start_time
    torch.save(model.state_dict(), f"{model_name}.pth")
    model.load_state_dict(torch.load(f"{model_name}.pth"))

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for step, (mel_spec, labels) in enumerate(tqdm(test_loader)):
            mel_spec = mel_spec.to(device)
            labels = labels.to(device)
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss /= len(test_loader)
    test_accuracy = correct / total
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}')

    result_df = pd.DataFrame({'True Label': all_labels, 'Predicted Label': all_preds})

    return train_val_results, total_time, result_df, test_loss, test_accuracy, test_f1

if __name__ == '__main__':
    folder_path = './FakeMusicCaps'  
    processed_folder = './porcess_dataset_cv2'
    train_dataset, val_dataset, test_dataset = prepare_datasets(folder_path, processed_folder)
    num_epochs = 10
    batch_size = 64
    lr = 0.001
    model_name_lists = ['VIT','CNN_LSTM', 'SeNet', 'mobileNet', 'vgg']#'ResNet'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers =4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers =4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers =4, pin_memory=True)
    for model_name in model_name_lists:
        if model_name == 'ResNet':
            model = ResNetClassifier(num_classes=2)
        if model_name == 'SeNet':
            model = SENetClassifier(num_classes=2)
        if model_name == 'mobileNet':
            model = MobileNetClassifier(num_classes=2)
        if model_name == 'vgg':
            model = VGGClassifier(num_classes=2)
        if model_name == 'VIT':
            model = ViTBinaryClassifier(num_classes=2)
        if model_name == 'CNN_LSTM':
            model = CNNLSTMClassifier(num_classes=2)

        results, total_time, test_result_df, test_loss, test_accuracy, test_f1 = train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, lr=lr, num_epochs = num_epochs)
        test_result_df.to_csv(f"{model_name}_test_results.csv", index=False)

        training_info = {
            'model_name': model_name,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': lr,
            'total_training_time': total_time,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1
        }

        with open(f'training_results_{model_name}.json', 'w') as json_file:
            json.dump({'results': results, 'info': training_info}, json_file, indent=4)


