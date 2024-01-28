import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Embedding
from pykeen.models import CompGCN
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

# PARAMETERS
dataset = "movielens"
emb_dim = 100
n_layers = 2
epochs = 15
drive_path = f"results/{dataset}/{emb_dim}"
# None if not using wiki2vec embeddings
setting = "iu"

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
    # with open(lab2id, "rb") as f:
    #   self.lab2id_dict = pickle.load(f)
    self.lab2id_dict = lab2id
  def __call__(self, ids):
    # translation: original id -> label -> pykeen id
    return [self.lab2id_dict[self.id2lab_dict[id]] for id in ids]

class Wiki2VecCompGCN(CompGCN):
  def __init__(self, wiki2vec_emb, **kwargs):
    super().__init__(**kwargs)
    num_embeddings = self.entity_representations[0].combined.entity_representations._embeddings.num_embeddings
    embedding_dim = self.entity_representations[0].combined.entity_representations._embeddings.embedding_dim
    emb = torch.randn((num_embeddings, embedding_dim))
    print(f"Total number of entities: {num_embeddings}")
    coverage = 0
    for id in wiki2vec_emb:
      try:
        emb[translate_id([id])[0],:] = wiki2vec_emb[id]
        coverage += 1
      except KeyError:
        # Item's id not present in the training data (hence in the pykeen id list)
        ...
    print(f"Number of covered entities: {coverage}")
    self.entity_representations[0].combined.entity_representations._embeddings = Embedding.from_pretrained(emb, freeze=False)

emb_training = TriplesFactory.from_path(
    f"results/{dataset}/pykeen_train.tsv",
    create_inverse_triples=True)

emb_testing = TriplesFactory.from_path(
    f"results/{dataset}/pykeen_test.tsv",
    entity_to_id=emb_training.entity_to_id,
    relation_to_id=emb_training.relation_to_id,
    create_inverse_triples=True)

translate_id = TranslateId(f"results/{dataset}/mapping_entities.tsv",
                           emb_training.entity_to_id)

if setting:
  # Prepare wiki2vec pre-trained embeddings
  wiki2vec_emb = torch.load(f"{drive_path}/wiki2vec_{emb_dim}_{setting}.pkl")
  result = pipeline(
      training=emb_training,
      testing=emb_testing,
      model=Wiki2VecCompGCN,
      model_kwargs=dict(wiki2vec_emb=wiki2vec_emb,
                        embedding_dim=emb_dim,
                        encoder_kwargs={"num_layers": n_layers}),
      evaluation_fallback = True,
      random_seed=4316,
      training_kwargs=dict(
          num_epochs=epochs,
      )
  )
else:
  result = pipeline(
      training=emb_training,
      testing=emb_testing,
      model="CompGCN",
      model_kwargs=dict(embedding_dim=emb_dim, encoder_kwargs={"num_layers": n_layers}),
      evaluation_fallback = True,
      random_seed=4316,
      training_kwargs=dict(
          num_epochs=epochs,
      ),
  )

model = result.model

def get_all_embeddings(model):
  combined = model.entity_representations[0].combined
  emb_e = combined.entity_representations().detach()
  emb_r = combined.relation_representations().detach()
  all_emb_e = []
  all_emb_r = []
  for layer in combined.layers:
    emb_e, emb_r = layer(emb_e, emb_r, combined.edge_index, combined.edge_type)
    all_emb_e.append(emb_e)
    all_emb_r.append(emb_r)
  return all_emb_e, all_emb_r

all_emb_e, all_emb_r = get_all_embeddings(model)

exp_name = f"{n_layers}_{setting}" if setting else f"{n_layers}"
torch.save(model, f"{drive_path}/{dataset}_model_{exp_name}.pt")
torch.save(all_emb_e, f"{drive_path}/{dataset}_embeddings_{exp_name}.pkl")
# save entity2id
with open(f"results/{dataset}/{dataset}_ent2id.pkl", "wb") as f:
  pickle.dump(emb_training.entity_to_id, f)