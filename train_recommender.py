import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from os import makedirs

# PARAMETERS
dataset = "dbbook"
batch_size = 128
epochs = 50
learning_rate = 0.001
# file containing the embeddings
embeddings_file = "embeddings.pkl"
concat = True

# Class to translate the dataset ids to pykeen ids
class TranslateId:
  def __init__(self, id2lab, lab2id):
    # original id -> label
    self.id2lab_dict = {}
    with open(id2lab) as f:
      for line in f:
        id, label = line.strip().split('\t')
        self.id2lab_dict[int(id)] = label
    # label -> pykeen id
    with open(lab2id, "rb") as f:
      self.lab2id_dict = pickle.load(f)
  def __call__(self, ids):
    # translation: original id -> label -> pykeen id
    return [self.lab2id_dict[self.id2lab_dict[id]] for id in ids]

class RecDataset(Dataset):
  def __init__(self, file):
    self.data = pd.read_csv(file, sep='\t', header=None)
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    user, item, like = self.data.loc[idx]
    return user, item, like

class UMAPSelfAttention(nn.Module):
  def __init__(self, emb_dim):
    super(UMAPSelfAttention, self).__init__()
    self.dense = nn.Linear(emb_dim * 2, emb_dim)
  def forward(self, emb):
    emb_1, emb_2 = emb
    cat = torch.cat([emb_1, emb_2], dim=1)
    att = F.softmax(self.dense(cat), dim=1)
    return att * emb_1 + (1 - att) * emb_2

class GATSelfAttention(nn.Module):
  def __init__(self, emb_dim):
    super(GATSelfAttention, self).__init__()
    self.weights = nn.Linear(emb_dim, emb_dim, bias=False)
    self.a_weight = nn.Linear(emb_dim * 2, 1, bias=False)
    self.leaky_relu = nn.LeakyReLU(0.2)
  def forward(self, emb):
    # emb shape -> (n_emb, batch_size, emb_dim)
    target_emb = emb[-1]

    target_emb = self.weights(target_emb).expand(emb.shape[0], -1, -1)
    other_emb = self.weights(emb)
    attn_coef = F.softmax(self.leaky_relu(self.a_weight(torch.cat([target_emb, other_emb], dim=2))), dim=0).squeeze(-1)
    return torch.einsum('ijk,ij->jk', emb, attn_coef)

class StandardAttention(nn.Module):
  def __init__(self, emb_dim):
    super(StandardAttention, self).__init__()
    self.weights = nn.Linear(emb_dim, 1, bias=False)
  def forward(self, emb):
    # emb shape -> (n_emb, batch_size, emb_dim)
    coef = self.weights(emb).squeeze(-1).softmax(0)
    return torch.einsum("ijk,ij->jk", emb, coef)

class MaxPool:
  def __init__(self, *args):
    pass
  def __call__(self, emb):
    return emb.max(0)[0]

class Concat:
  def __init__(self, *args):
    pass
  def __call__(self, emb):
    return emb.swapaxes(0,1).reshape((emb.shape[1],-1))

class RecommenderSelfAttention(nn.Module):
  def __init__(self, emb_list, attn_layer="UMAP"):
    super(RecommenderSelfAttention, self).__init__()
    emb_dim = emb_list[0].shape[1]
    if attn_layer == "UMAP":
      attn_class = UMAPSelfAttention
    elif attn_layer == "GAT":
      attn_class = GATSelfAttention
    elif attn_layer == "attn":
      attn_class = StandardAttention
    elif attn_layer == "maxpool":
      attn_class = MaxPool
    elif attn_layer == "concat":
      attn_class = Concat
      emb_dim = emb_dim * len(emb_list)
    else:
      raise ValueError("Wrong aggregation function name")
    hid_dim = emb_dim // 4
    self.emb_list = [nn.Embedding.from_pretrained(emb, freeze=True) for emb in emb_list]
    self.user_att = attn_class(emb_dim)
    self.item_att = attn_class(emb_dim)
    self.l1 = nn.Linear(emb_dim, hid_dim)
    self.l2 = nn.Linear(emb_dim, hid_dim)
    self.out = nn.Linear(hid_dim * 2, 1)
  def forward(self, users_id, items_id):
    users_all = torch.stack([emb(users_id) for emb in self.emb_list], dim=0)
    items_all = torch.stack([emb(items_id) for emb in self.emb_list], dim=0)
    users = self.user_att(users_all)
    items = self.item_att(items_all)
    x1 = F.relu(self.l1(users))
    x2 = F.relu(self.l2(items))
    x3 = torch.cat([x1, x2], dim=1)
    return F.sigmoid(self.out(x3))

class Recommender1(nn.Module):
  def __init__(self, emb):
    super(Recommender, self).__init__()
    emb_dim = emb.shape[1]
    hid_dim_0 = emb_dim // 2
    hid_dim_1 = emb_dim // 4
    self.emb = nn.Embedding.from_pretrained(emb, freeze=True) # Embeddings are not changed
    self.l1 = nn.Linear(emb_dim, hid_dim_0)
    self.l2 = nn.Linear(emb_dim, hid_dim_0)
    self.l3 = nn.Linear(hid_dim_0 * 2, hid_dim_0)
    self.l4 = nn.Linear(hid_dim_0, hid_dim_1)
    # self.out = nn.Linear(hid_dim_0 * 2, 1)
    self.out = nn.Linear(hid_dim_1, 1)
  def forward(self, users_id, items_id):
    users = self.emb(users_id)
    items = self.emb(items_id)
    x1 = F.relu(self.l1(users))
    x2 = F.relu(self.l2(items))
    x3 = torch.cat([x1, x2], dim=1)

    x3 = F.relu(self.l3(x3))
    x3 = F.relu(self.l4(x3))

    return F.sigmoid(self.out(x3))

class Recommender2(nn.Module):
  def __init__(self, emb):
    super(Recommender, self).__init__()
    emb_dim = emb.shape[1]
    hid_dim_0 = emb_dim // 4
    self.emb = nn.Embedding.from_pretrained(emb, freeze=True) # Embeddings are not changed
    self.l1 = nn.Linear(emb_dim, hid_dim_0)
    self.l2 = nn.Linear(emb_dim, hid_dim_0)
    self.out = nn.Linear(hid_dim_0 * 2, 1)
  def forward(self, users_id, items_id):
    users = self.emb(users_id)
    items = self.emb(items_id)
    x1 = F.relu(self.l1(users))
    x2 = F.relu(self.l2(items))
    x3 = torch.cat([x1, x2], dim=1)

    return F.sigmoid(self.out(x3))

train_data = RecDataset(f"results/{dataset}/train.tsv")
translate_id = TranslateId(f"results/{dataset}/mapping_entities.tsv",
                           f"results/{dataset}/{dataset}_ent2id.pkl")


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

loss_f = nn.BCELoss()
emb = torch.load(embeddings_file, map_location=dev)
# Training with only one embedding per entity
if concat:
    # Concatenate the embeddings from different layers
    emb = torch.cat(emb, dim=1).detach()
else:
    # Take only the embedding from the last layer
    emb = emb[-1]

print(emb.shape[1])
torch.manual_seed(4316)
if dataset == "movielens":
  model = Recommender1(emb)
elif dataset == "dbbook":
  model = Recommender2(emb)

# Other models
# model = RecommenderSelfAttention(emb, "attn")
model.to(dev)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
test_data = pd.read_csv(f"results/{dataset}/test.tsv", sep='\t', header=None)

makedirs(f"output_predictions", exist_ok=True)
makedirs(f"output_predictions/top5", exist_ok=True)
makedirs(f"output_predictions/top10", exist_ok=True)

for epoch in range(1, epochs + 1):
  model.train()
  mean_loss = []
  print(f"Starting epoch {epoch}")
  for users, items, target in tqdm(DataLoader(train_data, batch_size=batch_size)):
    opt.zero_grad()
    pred = model(torch.tensor(translate_id(users.tolist()), device=dev),
                 torch.tensor(translate_id(items.tolist()), device=dev))
    loss = loss_f(pred.squeeze(), target.to(dtype=torch.float32, device=dev).squeeze())
    mean_loss.append(loss.detach().cpu())
    loss.backward()
    opt.step()
  mean_loss = sum(mean_loss) / len(mean_loss)
  print(f"Finished epoch {epoch} with loss {mean_loss}")

  if epoch <= 10 or epoch % 5 == 0:
    print(f"Generating evaluations for epoch {epoch}")
    # Model evaluation
    model.eval()
    with torch.no_grad():
      top_5 = []
      top_10 = []

      # Group by user
      for user, ratings in test_data.groupby(0):
        # Original ids
        rated_items = torch.tensor(ratings[1].values)
        # Pykeen ids
        rated_items_id = torch.tensor(translate_id(rated_items.tolist()), device=dev)
        user_id = torch.tensor(translate_id([user]), device=dev).expand(len(rated_items_id))
        scores = model(user_id, rated_items_id).squeeze().cpu()
        top_scores, top_items_idx = scores.sort(descending=True)
        top10_scores, top10_items = top_scores[:10].tolist(), rated_items[top_items_idx[:10]].tolist()
        for i, (item, score) in enumerate(zip(top10_items, top10_scores)):
          if i < 5:
            top_5.append([user, item, score])
          top_10.append([user, item, score])

    pd.DataFrame(top_5).to_csv(f"output_predictions/top5/top5_epoch{epoch}.tsv", sep="\t", index=False, header=None)
    pd.DataFrame(top_10).to_csv(f"output_predictions/top10/top10_epoch{epoch}.tsv", sep="\t", index=False, header=None)
