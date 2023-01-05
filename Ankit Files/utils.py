from torch.utils.data import Dataset
import torch
import os
from tqdm import tqdm

from sklearn.metrics import accuracy_score

from transformers import RobertaModel


# Import tensorboard for logging
from torch.utils.tensorboard import SummaryWriter
writer_path = os.path.join('./tensorboard_logs')
writer = SummaryWriter(writer_path)


# Create dataset class
class MSMLDataset(Dataset):
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


#####################################################################################################################
#                                   Classifier Model class for MSML_641_Project                                     #
#####################################################################################################################

# Define HybridModel

class HybridModel(torch.nn.Module):
    
    def __init__(self, num_labels, numerical_features_size, transformer_features_size=32, dropout=0.2):
        super(HybridModel, self).__init__()
        self.transformer = RobertaModel.from_pretrained('roberta-base', return_dict=False)
        # transformer = BertModel.from_pretrained('bert-base-uncased', num_labels=2, output_hidden_states=False, output_attentions=False)
        # for param in self.model.parameters():
        #     param.requires_grad = True

        # self.dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(dropout)
        # self.fc1 = torch.nn.Linear(768, transformer_features_size)
        self.fc2 = torch.nn.Linear(768+numerical_features_size, num_labels)
        self.softmax = torch.nn.Softmax(dim=1)
        # self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask, numerical_features):

        # The pooled out is all as we want as it consolidates the features
        hidden_state_output,pooled_output = self.transformer(
            input_ids,
            attention_mask
        )
        droupout = self.dropout(pooled_output)
        # logits = self.fc1(droupout)

        # Concatenate numerical features along 2nd axis
        concat_layer = torch.cat((droupout, numerical_features), dim=1)

        logits = self.fc2(concat_layer)
        probs = self.softmax(logits)

        return probs

class TransformerModel(torch.nn.Module):
    def __init__(self, num_labels, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.transformer = RobertaModel.from_pretrained('roberta-base', return_dict=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(768, num_labels)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        hidden_state_output, pooled_output = self.transformer(
            input_ids,
            attention_mask
        )
        droupout = self.dropout(pooled_output)
        fc = self.fc(droupout)
        logits = self.softmax(fc)

        # Generate Argmax as predicitons
        predictions = torch.argmax(logits, dim=1)

        return logits,predictions

        

##################################################################################################################
#                                      Training Loop for MSML_641_Project                                       #
##################################################################################################################

# Training Loop
def train_hybrid(model, train_dataloader, val_dataloader, criterion, optimizer, learning_rate_scheduler, num_epochs, device):
    # Set device
    model.to(device)

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        
        
        ######## TRAINING #########        

        # Set model to training mode
        model.train()

        # Set traingin loss and accuracy to 0
        train_accuracy = 0
        train_loss = 0

        for data in train_dataloader:

            # Set the gradients to zero for each batch
            model.zero_grad()

            # Set the input and labels to the device by getting to right dimension
            input_ids= data['input_ids'].squeeze(1).to(device)
            attention_mask = data['attention_mask'].to(device)
            numerical_features = data['numerical_features'].to(device).float()

            labels = data['labels'].to(device)

            # Forward pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features
            )

            # Compute the loss
            batch_loss = criterion(logits, labels)
            
            # Add loss to total loss
            train_loss += batch_loss.item()

            # Backward pass
            batch_loss.backward()

            # Clip the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights
            optimizer.step()

            # Update scheduler
            learning_rate_scheduler.step()

            # Get the training accuracy
            _, predicted = torch.max(logits.data, 1)
            # Calculate the accuracy
            # train_accuracy += (predicted == labels).sum().item()/len(labels)
            # calculate the accuracy using sklearn
            train_accuracy += accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())



        # Calculate average loss and accuracy
        avg_loss = train_loss / len(train_dataloader)
        avg_accuracy = train_accuracy / len(train_dataloader)

        # Add to tensorboard
        writer.add_scalar('train/Loss', avg_loss, epoch)
        writer.add_scalar('train/Accuracy', avg_accuracy, epoch)

        # Add loss and accuracy to lists
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)


        ######## VALIDATION #########

        # Set model to evaluation mode
        model.eval()

        # Calculate validation accuracy and loss to 0
        val_accuracy = 0
        val_loss = 0
        
        for data in val_dataloader:
            
            # Set the input and labels to the device by getting to right dimension
            input_ids= data['input_ids'].squeeze(1).to(device)
            attention_mask = data['attention_mask'].to(device)
            numerical_features = data['numerical_features'].to(device).float()
            labels = data['labels'].to(device)

            with torch.no_grad():
                # Forward pass
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features
                )

                # Compute the loss
                batch_loss = criterion(logits, labels)

                # Add loss to total loss
                val_loss += batch_loss.item()

                _, predicted = torch.max(logits.data, 1)
                acc = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
                # print(acc)
                val_accuracy += acc
            
        # Calculate average loss and accuracy
        avg_loss = val_loss / len(val_dataloader)
        avg_val_accuracy = val_accuracy / len(val_dataloader)

        # Add to tensorboard
        writer.add_scalar('valid/Loss', avg_loss, epoch)
        writer.add_scalar('valid/Accuracy', avg_val_accuracy, epoch)

        # Add validation accuracy and loss to lists
        val_accuracies.append(avg_val_accuracy)
        val_losses.append(avg_loss)

    # Print the results
    print(f'Training Loss: {avg_loss}')
    print(f'Training Accuracy: {avg_accuracy}')
    print(f'Validation Accuracy: {avg_val_accuracy}')
    print('\n')

    # Return the model and the lists
    return model, train_accuracies, train_losses, val_accuracies, val_losses                                                            