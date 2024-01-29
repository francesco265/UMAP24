import torch
import pickle
from tqdm import tqdm
from wikipedia2vec import Wikipedia2Vec

# PARAMETERS
# data_to_wikiname.py output file
id2wikiname_file = "id2wikiname.pkl"
# Set training_file to None if you don't want to include user embeddings
training_file = "results/dbbook/train.tsv"
# Wikipedia2Vec dump file
wiki2vec_dump = "enwiki_20180420_100d.pkl"

id2wikiname = torch.load(id2wikiname_file)
wiki2vec = Wikipedia2Vec.load(wiki2vec_dump)

found = set()
not_found = set()
not_found_name = []
embeddings = {}
for id in tqdm(id2wikiname):
	ok = False
	entity = id2wikiname[id]
	if type(entity) is list:
		emb = torch.zeros((100,))
		tot = 0
		for e in entity:
			res = wiki2vec.get_entity(e)
			if res:
				emb += torch.from_numpy(wiki2vec.get_entity_vector(e))
				tot += 1
		if tot > 0:
			emb /= tot
			ok = True
	else:
		res = wiki2vec.get_entity(entity)
		if res:
			emb = torch.from_numpy(wiki2vec.get_entity_vector(entity))
			ok = True

	if ok:
		found.add(id)
		embeddings[id] = emb
	else:
		not_found.add(id)
		not_found_name.append(entity)

print(f"Items found: {len(found)}")
print(f"Items not found: {len(not_found)}")

# Users
if training_file:
  f = open(training_file)
  likes = {}
  for line in f:
    user, item, rel = map(lambda x: int(x.strip()), line.split("\t"))
    if rel == 1:
      user = int(user)
      if user in likes:
        likes[user].append(int(item))
      else:
        likes[user] = [item]
  f.close()

  users_emb = {}
  for user in likes:
    user_emb = torch.zeros((100,))
    n_items = 0
    for item in likes[user]:
      if item in embeddings:
        user_emb += embeddings[item]
        n_items += 1
    if n_items > 0:
      users_emb[user] = user_emb / n_items

  # Merge user and item dicts
  embeddings.update(users_emb)

print(f"Final number of embeddings: {len(embeddings)}")
torch.save(embeddings, "wiki2vec_embeddings.pkl")
