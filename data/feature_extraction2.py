import os
import pickle
import argparse

import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import load_qmnist_data

torch.backends.cudnn.deterministic = True
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)


transform = {
    'nist': transforms.Compose(
        [lambda x: Image.fromarray(x).convert("RGB"),
        transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
}

class QMNISTImages(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

def get_features(model, images_array, device, transform=None, batch_size=128):
    dataset = QMNISTImages(images_array, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    x_tab = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader):
            data = data.to(device)
            features = model.get_features(data)
            x_tab.append(features.cpu().numpy())
    return np.vstack(x_tab)

def test(model, target_test_loader, device):
    model.eval()
    correct = 0    
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(device), target.to(device)
            s_output = model.predict(data)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    return acc

class TransferNet(nn.Module):
    def __init__(self, num_class, transfer_loss='mmd', 
        use_bottleneck=True, bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = ResNetBackbone()
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()
        
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        self.adapt_loss = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, source, target, source_label):
        source = self.base_network(source)
        target = self.base_network(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer
        kwargs = {}
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        clf = self.classifier_layer(x)
        return clf

    def get_features(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        return x

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass
        
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def output_num(self):
        return self._feature_dim

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract features from QMNIST images using Resnet50 pretrained with Fake-MNIST')
    parser.add_argument('--model_path', help='Path for loading model weights', default=None)
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Load QMNIST data from QMNIST.pickle
    pickle_file = os.path.join(current_dir, 'QMNIST_ppml.pickle')
    x_defender, x_reserve, y_defender, y_reserve = load_qmnist_data(pickle_file)
    print('Data loaded.')
    
    # Load pre-trained resnet model to preprocess
    if args.model_path is None:
        model_file_name = os.path.join(current_dir,"resnet50_amber-salad-1.pth")
    else:
        model_file_name = args.model_path

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print('Using {} as device.'.format(device))

    model = TransferNet(10)
    model.to(device)
    model.load_state_dict(torch.load(model_file_name, map_location=device))

    # Transform image data into tabular data using pre-trained NN

    x_defender_tabular = get_features(model, x_defender, device, transform["nist"])
    x_reserve_tabular = get_features(model, x_reserve, device, transform["nist"])

    # Store the results in a .pickle
    split_dict = dict()
    split_dict['x_defender'] = x_defender_tabular
    split_dict['x_reserve'] = x_reserve_tabular
    split_dict['y_defender'] = y_defender
    split_dict['y_reserve'] = y_reserve
    # Store the dict using pickle
    with open(os.path.join(current_dir, 'QMNIST_tabular2_ppml.pickle'), 'wb') as f:
        pickle.dump(split_dict, f)

    print('Tabular data stored.')