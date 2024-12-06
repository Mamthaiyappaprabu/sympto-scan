# SYMPTO SCAN 

### INTRODUCTION :
The project focuses on developing an AI-powered system for accurate lung cancer detection from medical images.
The AI solution aims to assist radiologists by providing faster, more reliable diagnoses, improving efficiency and patient care.

### STATEMENT OF PROBLEM :
* Diagnosing lung cancer involves analyzing a high volume of medical images, leading to potential human error and oversight, especially in early-stage cases.

* Fatigue and workload can result in missed or delayed diagnoses, negatively impacting patient outcomes and survival rates.
## HARDWARE AND SOFTWARE :

* GPU: NVIDIA RTX 3090 or A100 
For faster training and processing of large CT scan datasets.
* CPU: Intel Core i9 or AMD Ryzen 9
Handles data preparation and general computations.
* RAM: 32GB or more
Ensures smooth processing of large datasets and model training.
* Storage: 1TB SSD + 4TB HDD
SSD for fast data access, HDD for storing raw and processed datasets.
* Programming Language: Python
Easy to use and supported by many AI tools.


### SCOPE OF THE PROJECT :
**Diagnosing lung cancer**

**Scope:**
* Focused on detecting the presence of cancer cells in lung scans.
* Aims to identify malignancies at an early stage for timely intervention.
* Reduces dependency on manual interpretation by radiologists.

**Target Audience:**

 Patients,Medical professionals, healthcare institutions.

**Deliverables:**
* An AI model capable of predicting the presence or absence of lung cancer with high accuracy.
* A web-based platform allowing users to upload lung scans and receive diagnostic results.
* A user-friendly interface for model deployment.
### FUTURE WORKS :
* Real-time Monitoring: The system could help doctors track a patient's health in real-time, predicting how the cancer might progress and helping doctors take early action.
  
* Personalized Treatment Plans: The AI could help create treatment plans tailored to each patient based on their unique health data, leading to better outcomes and fewer side effects.



### REQUIREMENTS OF THE PROJECT :
**Data Requirements:**

* Access to high-quality, labeled lung scan datasets (CT scans, X-rays).
* Image preprocessing for quality enhancement (noise reduction, normalization).
  
**AI Model Development:**

* Select deep learning algorithms (e.g., CNNs) for image classification.
* Train the model using labeled datasets and validate performance with cross-validation techniques.
* Achieve high accuracy in early-stage cancer detection.
  
**Platform Development:**

* Develop a user-friendly web interface for uploading scans and receiving results.
* Implement backend architecture for image processing and model inference.
* Deploy on scalable cloud infrastructure to ensure reliability and accessibility.
  
**Security and Privacy:**

* Ensure patient data protection with strong encryption and secure uploads.
* Comply with healthcare privacy regulations (HIPAA, GDPR).
  
**Testing and Evaluation:**

* Validate model accuracy with unseen datasets.
* Conduct user testing for platform usability and effectiveness.
* Continuously improve the model based on feedback.
  
**Compliance:**

* Adhere to healthcare industry regulations and standards for medical data handling.
### MODEL AND IMPLEMENTATION  :

**DenseNet**
* DenseNet connects each layer to every other layer in a feed-forward fashion, ensuring efficient feature reuse and alleviating the vanishing gradient problem.
* Extracts hierarchical features from medical images for classification.
* Enhances gradient flow and encourages feature propagation.
* Mitigates vanishing gradients and improves the flow of information and gradients across the network.
* Suitable for tasks requiring high precision, such as lung cancer detection.



  ### ARCHITECTURE DIAGRAM :

  ![image](https://github.com/user-attachments/assets/b164d62c-198d-483b-9c44-015673c0b3f8)

### PROGRAM :
```Python
# Import necessary libraries
from google.colab import drive
drive.mount('/content/drive')
import os
import random
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

# Data directory
data_dir = '/content/drive/MyDrive/Doodle/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset'

# Categories in the dataset
Name1=['Normal cases', 'Bengin cases', 'Malignant cases']
Name = sorted(Name1)

# Mapping category names to numeric labels
normal_mapping = dict(zip(Name, range(len(Name))))
reverse_mapping = dict(zip(range(len(Name)), Name))

# Load images into dataset
dataset = []
for i in tqdm(range(len(Name))):
    path = os.path.join(data_dir, Name[i])
    for im in os.listdir(path):
        img_path = os.path.join(path, im)
        img1 = cv2.imread(img_path)
        if img1 is None:
            continue
        img2 = cv2.resize(img1, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        img3 = img2.astype(np.float32)
        image = torch.from_numpy(img3).permute(2, 0, 1)
        labeli = normal_mapping[Name[i]]
        dataset += [[image, labeli]]

# Function to display an image
def show_image(img, label):
    img2 = img.permute(1, 2, 0).numpy().astype(int)
    plt.imshow(img2)
    plt.title(reverse_mapping[label])
    plt.show()

# Split the dataset into train, validation, and test sets
torch.manual_seed(20)
val_size = len(dataset) // 10
test_size = len(dataset) // 5
train_size = len(dataset) - val_size - test_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

# DataLoader for batching
batch_size = 32
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size, num_workers=2, pin_memory=True)

# Function to calculate accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Base model class
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

# Update the model to DenseNet-121
class DenseNetModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.densenet121(pretrained=True)
        num_ftrs = self.network.classifier.in_features
        self.network.classifier = nn.Linear(num_ftrs, 3)  # 3 classes (Normal, Benign, Malignant)

    def forward(self, xb):
        return self.network(xb)

# Training and evaluation functions
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# Move data and model to GPU if available
def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader(DataLoader):
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

device = get_default_device()
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
test_loader = DeviceDataLoader(test_loader, device)

# Initialize and train the DenseNet model
model = DenseNetModel().to(device)
epochs = 100
lr = 0.001
history = fit(epochs, lr, model, train_loader, val_loader)

# Evaluate on test set
test_result = evaluate(model, test_loader)
print(f"Test accuracy: {test_result['val_acc']:.4f}")

# Plot the training and validation accuracies and losses
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.show()

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss over Epochs')
    plt.show()

plot_accuracies(history)
plot_losses(history)

```

### OUTPUT :


![Screenshot 2024-12-05 204406](https://github.com/user-attachments/assets/fc6826fb-f3f6-4705-97b0-69bca04a58db)


![Screenshot 2024-12-05 204517](https://github.com/user-attachments/assets/b52ee478-aa74-463b-816e-9a8b378de868)

* Accuracy: 0.9911%
* Precision: 1.00%
* Recall: 0.97%

### RESULT :
* Accurate and Efficient Lung Cancer Detection:
   The AI system improves diagnostic accuracy and speed, enabling early detection and timely 
   intervention for lung cancer.
* Scalable Healthcare Solution:
   The project provides a foundation for broader AI applications in medical imaging, 
  supporting radiologists and improving patient outcomes across various conditions.







  



