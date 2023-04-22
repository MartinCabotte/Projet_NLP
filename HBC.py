from transformers import AutoTokenizer, logging, DistilBertModel, RobertaModel
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.set_verbosity_error()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


print("--- Chargement des datasets ---")


#On donne les informations du dataset
directory = "dataset/"
train_name = "train.En.csv"
test_name = "task_A_En_test.csv"


#On défini les paramètres du dataloader
batch_size = 20
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1}


#On créé nos données de test
panda_Train = pd.read_csv(directory+train_name, index_col=[0])
panda_Train = panda_Train.drop(labels="rephrase",axis=1)
X_train = panda_Train['tweet']
y_train = panda_Train.iloc[:,1:]

cpt = 0
for phrase in X_train:
    if pd.isna(phrase):
        print("ERROR - il y a des NAN dans le dataset")
        print("--- Traitement en cours ---")
    cpt += 1

indsNan = []
cpt = 0
for phrase in X_train:
    if pd.isna(phrase):
        indsNan.append(cpt)
    cpt += 1
    
X_train = X_train.drop(indsNan)
y_train = y_train.drop(indsNan)

# https://stackoverflow.com/questions/67312321/how-to-remove-urls-between-texts-in-pandas-dataframe-rows
X_train = X_train.str.replace(r' https:\/\/.*', ' ', regex=True).str.strip()

cpt = 0
for phrase in X_train:
    if pd.isna(phrase):
        print("ERROR - il y a encore des NAN dans le dataset")
    cpt += 1

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

X_train = tokenizer(X_train.to_list(), return_tensors="pt", padding=True, truncation=True, max_length=128)

train_inputs = X_train['input_ids']
train_masks  = X_train['attention_mask']
train_labels = torch.tensor(np.asarray(y_train))

#On créé notre dataset pyTorch
train_iSarcasm = TensorDataset(train_inputs,train_masks,train_labels)
training_generator = DataLoader(train_iSarcasm, **params)


#On regarde que tout soit bien chargé
for i, batch in enumerate(training_generator):
    if i > 0:
        break
    print("Batch size : "+str(len(batch[0])))
    
#On va créer nos données de validation
panda_Test = pd.read_csv(directory+test_name, index_col=[0])
X_test = panda_Test.index
y_test = panda_Test['sarcastic']

X_test = tokenizer(X_test.to_list(), return_tensors="pt", padding=True, truncation=True, max_length=128)

test_inputs = X_test['input_ids']
test_masks  = X_test['attention_mask']
test_labels = torch.tensor(np.asarray(y_test))

#On créé notre dataset pyTorch
test_iSarcasm = TensorDataset(test_inputs,test_masks,test_labels)
test_generator = DataLoader(test_iSarcasm, **params)

#On regarde que tout soit bien chargé
for i, batch in enumerate(test_generator):
    if i > 0:
        break
    print("Batch size : "+str(len(batch[0])))
    
print("--- Chargement des datasets terminé ---")


class HeuristicBertClassifier(nn.Module):
    def __init__(self):
        super(HeuristicBertClassifier, self).__init__()

        self.bert = RobertaModel.from_pretrained("roberta-base")
        self.tronc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU()
        )
        self.h = nn.Linear(64, 6)
        self.s = nn.Linear(70,1)

    def forward(self, text, mask):
        x = self.bert(input_ids=text, attention_mask=mask, output_attentions=False, output_hidden_states=False).last_hidden_state[:,0,:] # [CLS]
        x = self.tronc(x)
        h = self.h(x)
        x = self.s(torch.cat((x,h),dim=1))
        return x, h
    
class SimpleBertClassifier(nn.Module):
    def __init__(self):
        super(HeuristicBertClassifier, self).__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.tronc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, text, mask):
        x = self.bert(input_ids=text, attention_mask=mask, output_attentions=False, output_hidden_states=False).last_hidden_state[:,0,:] # [CLS]
        x = self.tronc(x)
        return x
    

model = HeuristicBertClassifier()
model.to(device)

# On va optimiser notre modèle
import torch.optim as optim

# On déclare notre loss
criterion = nn.BCEWithLogitsLoss()

# On se déclare un optimiseur qui effectuera la descente de gradient
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# L'historique pour print plus tard
loss_history, train_accuracy_history, valid_accuracy_history = [], [], []
n_epoch = 3

# On réalise notre nombre d'epochs
for epoch in range(n_epoch):
    
    running_loss = 0
    # On loop sur le batch
    for i, batch in enumerate(training_generator):
        
        batch = (i.to(device) for i in batch)
        sequences, masks, labels = batch
        sarcastic = labels[:,0].unsqueeze(1)
        categories = labels[:,1:]

        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        out_sarcastic, out_categories = model(sequences, masks)

        loss = criterion(out_sarcastic, sarcastic) 
        if not torch.isnan(categories).any():
            loss += criterion(out_categories, categories)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    
        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:  
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0                                              
            
print("--- Entrainement terminé ---")

print("--- Phase de test ---")

preds = []
truth = []

with torch.no_grad():
    for i, batch in enumerate(test_generator):
            
        batch = (i.to(device) for i in batch)
        sequences, masks, labels = batch

        # forward + backward + optimize
        out_sarcastic, _ = model(sequences, masks)

        # collect predictions and labels
        preds.append(out_sarcastic.cpu())
        truth.append(labels.cpu())

preds = torch.cat(preds, dim=0).squeeze()
truth = torch.cat(truth, dim=0).squeeze()

acc = BinaryAccuracy()
f1 = BinaryF1Score()

accuracy = acc(preds,truth)
f1score = f1(preds,truth)

print(f'Final Accuracy: {accuracy}')
print(f'Final F1-Score: {f1score}')





