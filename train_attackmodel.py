import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import util

class Attack_M(torch.nn.Module): 
    def __init__(self, feat_dim):
        super(Attack_M, self).__init__()
        self.fc1 = nn.Linear(feat_dim,512) ## ViT-B-32 : 1024 / ViT-B-16 : 1024 / RN 50 : 2048
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,2)        
        self.relu = torch.nn.ReLU() # instead of Heaviside step fn
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x) # instead of Heaviside step fn
        x = self.fc2(x)
        x = self.relu(x) # instead of Heaviside step fn    
        x = self.fc3(x)
        x = self.relu(x) # instead of Heaviside step fn            
        output = self.fc4(x) # instead of Heaviside step fn        
        return output


class Baseline_M(torch.nn.Module): 
    def __init__(self, feat_dim):
        super(Baseline_M, self).__init__()
        self.fc1 = nn.Linear(feat_dim,512) ## ViT-B-32 : 1024 / ViT-B-16 : 1024 / RN 50 : 2048
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,2)        
        self.relu = torch.nn.ReLU() # instead of Heaviside step fn
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x) # instead of Heaviside step fn
        x = self.fc2(x)
        x = self.relu(x) # instead of Heaviside step fn    
        x = self.fc3(x)
        x = self.relu(x) # instead of Heaviside step fn            
        output = self.fc4(x) # instead of Heaviside step fn        
        return output

# Define the weight orthogonality regularization
def weight_orthogonality_regularizer(model, lambda_reg):
    ortho_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            W = param.view(param.size(0), -1)
            WT_W = torch.matmul(torch.transpose(W, 0, 1), W)
            ortho_loss += torch.norm(WT_W - torch.eye(WT_W.size(0)).to(W.device))
    ortho_loss *= lambda_reg
    return ortho_loss


def train_attackmodel(args, selected_t_feat_lst_tar, selected_nt_feat_lst_tar, true_train, pseudo_train, train_threshold, device):
    
    X1 = torch.stack(selected_t_feat_lst_tar) ## pseudo-train
    X2 = torch.stack(selected_nt_feat_lst_tar) 
    
    ### Sensitivity to Nontrain size 
    if args.nt_length == 3000 and args.t_length == 7000:
        min_num = 5000    
    elif args.nt_length == 7000 and args.t_length == 10000:
        min_num = 5000        
    elif args.nt_length == 15000 and args.t_length == 15000:
        min_num = 10000  
    elif args.nt_length == 15000 and args.t_length == 30000:
        min_num = 9000          
    elif args.nt_length == 20000 and args.t_length == 30000:
        min_num = 30000
    elif args.nt_length == 20000 and args.t_length == 50000:
        min_num = 30000     
    elif args.nt_length == 20000 and args.t_length == 100000:
        min_num = 30000      
    elif args.nt_length == 20000 and args.t_length == 150000:
        min_num = 30000          
    elif args.nt_length == 30000 and args.t_length == 50000:
        min_num = 50000
    elif args.nt_length == 30000 and args.t_length == 60000:
        min_num = 70000
    elif args.nt_length == 40000 and args.t_length == 70000:
        min_num = 89000
    else:
        min_num = min(len(selected_t_feat_lst_tar), len(selected_nt_feat_lst_tar))
        
    ################################################## random sampling
    X1_choice = np.random.choice(np.arange(len(X1)), size=min_num, replace=False) ## pseudo-train
    X2_choice = np.random.choice(np.arange(len(X2)), size=min_num, replace=False)    
    ################################################## mislabel rate 
    pseudo_t = np.arange(true_train, len(selected_t_feat_lst_tar)) ## pseudo-train
    pseudo_t_ind = np.intersect1d(pseudo_t, X1_choice)  ## selected_pseudo-train
    mis_rate = len(pseudo_t_ind)/(len(X1_choice)+len(X2_choice))
    ##################################################
    X1 = X1[X1_choice] 
    X2 = X2[X2_choice]
    
    Y1 = torch.ones(len(X1)).to(dtype = torch.long)
    Y2 = torch.zeros(len(X2)).to(dtype = torch.long)
    ##################################################    
    if args.model == "ViT-B-32":
        feat_dim = 1024
    elif args.model == "ViT-B-16":
        feat_dim = 1024
    elif args.model == "ViT-L-14":
        feat_dim = 1536
    elif args.model == "RN50":
        feat_dim = 2048
    elif args.model == "RN101":
        feat_dim = 1024
        
    attack_model = Attack_M(feat_dim).to(device)

    learning_rate = 0.001
    batch_size_attack = 32
    num_epochs = 15

    data = torch.cat( [X1 , X2] ) 
    labels = torch.cat( [Y1 , Y2] ) 
    
    print(util.magenta(f"To train an attack model, Member shape is : {X1.shape}"))
    print(util.magenta(f"To train an attack model, Non-member shape is : {X2.shape}"))
    print(util.magenta(f"To train an attack model, Data shape is : {data.shape}"))
    print(util.magenta(f"To train an attack model, Label shape is : {labels.shape}"))

    dataset = torch.utils.data.TensorDataset(data, labels)

    # Split the dataset into a train and test set using the `random_split` method
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Check the size of the train and test sets
    print(util.magenta(f"Size of train set : {len(train_dataset)}"))
    print(util.magenta(f"Size of test set : {len(test_dataset)}"))
    
    # Create DataLoaders for the train and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size_attack, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_attack, shuffle=False)

    # Initialize the model, optimizer, and criterion
    optimizer = optim.Adam(attack_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    patience = 3
    best_val_loss = float('inf')
    no_improvement_epochs = 0

    # Train the model
    for epoch in range(num_epochs):
        attack_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = attack_model(inputs)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
        print(util.magenta(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}"))
          
        # Test the model    
        attack_model.eval()
        val_loss = 0.0    
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)        
                outputs = attack_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_loss += criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= total
        print(util.magenta(f"Attack Model Test Accuracy is : {correct/total:.4f}"))
                  
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f'Early stopping after {epoch} epochs')
            break
    
    return attack_model, mis_rate

