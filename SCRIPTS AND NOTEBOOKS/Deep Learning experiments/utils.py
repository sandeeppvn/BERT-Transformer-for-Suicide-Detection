from torch.utils.data import Dataset
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

from transformers import RobertaModel


# Import tensorboard for logging
from torch.utils.tensorboard import SummaryWriter
writer_path = os.path.join('./tensorboard_logs')
writer = SummaryWriter(writer_path)


# Create dataset class
class HybridDataset(Dataset):
    def __init__(self, encodings, labels, numerical_features):
        self.encodings = encodings
        self.labels = labels
        self.numerical_features = numerical_features

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            k: v[idx].clone().detach() for k, v in self.encodings.items()
        }
        item['labels'] = torch.tensor(int(self.labels.iloc[idx]))

        if self.numerical_features is not None:
            item['numerical_features'] = torch.tensor(self.numerical_features.iloc[idx])

        return item


# Create dataset class
class TransformerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            k: v[idx].clone().detach() for k, v in self.encodings.items()
        }
        item['labels'] = torch.tensor(int(self.labels.iloc[idx]))

        return item

class NumericalDataset(Dataset):
    def __init__(self, numerical_features, labels):
        self.numerical_features = numerical_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            'numerical_features': torch.tensor(self.numerical_features.iloc[idx]),
            'labels': torch.tensor(int(self.labels.iloc[idx]))
        }

        return item


#####################################################################################################################
#                                   Classifier Model class for MSML_641_Project                                     #
#####################################################################################################################

# Define HybridModel
class HybridModel_v1(torch.nn.Module):
    
    def __init__(self, num_labels, transformer_model, numerical_features_size, transformer_features_size=32):
        super(HybridModel_v1, self).__init__()
        self.transformer = transformer_model
        # Freeze Transformer weights
        for param in self.transformer.parameters():
            param.requires_grad = False

        self.fc1 = torch.nn.Linear(768, transformer_features_size)
        self.fc2 = torch.nn.Linear(transformer_features_size+numerical_features_size, num_labels)
        self.softmax = torch.nn.Softmax(dim=1)
        # self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask, numerical_features):

        # The pooled out is all as we want as it consolidates the features
        fc = self.transformer(
            input_ids,
            attention_mask
        )
        fc1 = self.fc1(fc)

        # Concatenate numerical features along 2nd axis
        concat_layer = torch.cat((fc1, numerical_features), dim=1)

        logits = self.fc2(concat_layer.float())
        logits = self.softmax(logits)

        predictions = torch.argmax(logits, dim=1)

        return logits, predictions

class LinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.swish = swish

    def forward(self, x):
        return self.swish(self.linear(x))

class TunedTransformerModel(torch.nn.Module):
    def __init__(self, transformer_model):
        super(TunedTransformerModel, self).__init__()
        self.transformer = RobertaModel.from_pretrained('roberta-base', return_dict=False)
        self.fc = transformer_model.fc1
        
    def forward(self, input_ids, attention_mask):
        hidden_state_output, pooled_output = self.transformer(
            input_ids,
            attention_mask
        )
        fc = self.fc(pooled_output)
        return fc

class TransformerModel(torch.nn.Module):
    def __init__(self, num_labels):
        super(TransformerModel, self).__init__()
        self.transformer = RobertaModel.from_pretrained('roberta-base', return_dict=False)
        for param in self.transformer.parameters():
            param.requires_grad = False

        self.fc1 = torch.nn.Linear(768, 768)
        self.fc2 = torch.nn.Linear(768, num_labels)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        hidden_state_output, pooled_output = self.transformer(
            input_ids,
            attention_mask
        )

        fc1 = self.fc1(pooled_output)
        fc2 = self.fc2(fc1)
        logits = self.softmax(fc2)

        # Generate Argmax as predicitons
        predictions = torch.argmax(logits, dim=1)

        return logits,predictions

def swish(x):
    return x * torch.sigmoid(x)


class NumericalModel(torch.nn.Module):
    def __init__(self, num_of_numerical_features, num_labels, dropout=0.2):
        super(NumericalModel, self).__init__()

        self.fc1 = torch.nn.Linear(num_of_numerical_features, 32)
        self.batchnorm1 = torch.nn.BatchNorm1d(32)

        self.fc2 = torch.nn.Linear(32, 16)
        self.batchnorm2 = torch.nn.BatchNorm1d(16)

        self.fc3 = torch.nn.Linear(16, num_labels)

        self.dropout = torch.nn.Dropout(dropout)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, numerical_features, labels):
        
        fc1 = swish(self.fc1(numerical_features))
        batchnorm1 = self.batchnorm1(fc1)

        fc2 = swish(self.fc2(batchnorm1))
        batchnorm2 = self.batchnorm2(fc2)

        fc3 = self.fc3(batchnorm2)
        droupout = self.dropout(fc3)

        logits = self.softmax(droupout)

        # Generate Argmax as predicitons
        predictions = torch.argmax(logits, dim=1)

        return logits,predictions
        
