# SPARQLWrapper, wikipedia-api are needed

import re
import requests
import pandas as pd
from tqdm import tqdm
from urllib.parse import quote, unquote
from SPARQLWrapper import SPARQLWrapper, JSON

# PARAMETERS
# tsv file containing (id, url) pairs,
# it should contain items, properties or both
input_file = "results/dbbook/dbbook_entities.tsv"

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

def make_request(entity: str):
  return requests.get(f"http://api.live.dbpedia.org/resource/en/{entity}",
                      headers={"Accept":"application/json"})

def make_sparql_request(entity_list: str):
  entity_list = [f"<{x}>" for x in entity_list]
  sparql.setQuery("""
    SELECT ?entity ?id WHERE
      {
        ?entity dbo:wikiPageID ?id .
        FILTER (?entity in (%s))
      }
  """ % (", ".join(entity_list)))
  return sparql.queryAndConvert()["results"]["bindings"]

def fix_unicode(s: str) -> str:
  bo = s.encode()
  idx = [i for i, e in enumerate(bo) if e == 195]
  if len(idx) == 0:
    return s
  new_s = bo[:idx[0] + 1]
  for i in range(len(idx) - 1):
    new_s += bytes(bo[idx[i] + 3:idx[i + 1] + 1])
  new_s += bo[idx[-1] + 3:]
  print(s)
  return new_s.decode()

data = pd.read_csv(input_file, sep="\t")
id_list = data.graphid.tolist()
# Unicode fix (only for dbbook's properties)
# dburl_list = data.dburl.map(lambda x: fix_unicode(x)).tolist()
dburl_list = data.dburl.tolist()
pageid_dict = {}
not_found = {}

for i in tqdm(range(0, len(id_list), 50)):
  ids = id_list[i:i+50]
  dburls = dburl_list[i:i+50]
  results = make_sparql_request(dburls)
  entities = [x["entity"]["value"] for x in results]
  for id, url in zip(ids, dburls):
    try:
      index = entities.index(url)
      pageid_dict[id] = results[index]["id"]["value"]
    except ValueError:
      not_found[url] = id

print(f"items found = {len(pageid_dict)}, items not found = {len(not_found)}")

# Try to solve not found items
print(f"Items found before: {len(pageid_dict)}")
keys = list(not_found.keys())
for i in range(0, len(keys), 50):
  urls_original = keys[i:i+50]
  urls = [unquote(x) for x in keys[i:i+50]]
  results = make_sparql_request(urls)
  for entry in results:
    entity_name = urls_original[urls.index(entry["entity"]["value"])]
    pageid_dict[not_found[entity_name]] = entry["id"]["value"]
    not_found.pop(entity_name)
print(f"Items found after: {len(pageid_dict)}")

id2wikiname = {}
# PageID to Wikipedia name
keys = list(pageid_dict.keys())
for i in range(0, len(keys), 50):
  ids = keys[i:i+50]
  pageids = [pageid_dict[x] for x in ids]
  response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids={'|'.join(pageids)}&format=json").json()
  query = response["query"]["pages"]
  for id, pid in zip(ids, pageids):
    try:
      id2wikiname[id] = query[pid]["title"]
    except KeyError:
      print(f"Item {id} is linked to the faulty wikiPageID {pid}")
      new_pid = input("Insert the correct value: ")
      if new_pid:
        new_response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids={new_pid}&format=json").json()
        new_query = new_response["query"]["pages"]
        id2wikiname[id] = new_query[new_pid]["title"]

# Fix categories

n_cat = 0
wiki = wikipediaapi.Wikipedia('UniBot/1.0', 'en')
for id in id2wikiname:
  wikipage = id2wikiname[id]
  if not "Category:" in wikipage:
    continue
  page = wiki.page(wikipage)
  members = [x.title() for x in page.categorymembers]
  id2wikiname[id] = members
  n_cat += 1

print(f"Total number of categories: {n_cat}")

torch.save(id2wikiname, "id2wikiname.pkl")