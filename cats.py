"""
Ryan Sheatsley
6/14/2019
"""


def assemble(database='database.pkl'):
  """
  Exhaustively compute CATS combinations from the parts database
  Field definitions:
  - body:   type, weapons, gadgets, wheels, health, energy, bonus
  - weapon: type, damage, energy, bonus
  - wheel:  type, health, bonus
  - gadget: type, health, energy, bonus
  """
  import itertools as it
  import numpy as np
  import pickle as pk

  # load the parts databse
  try:
    with open(database, 'rb') as f:
      db = pk.load(f)
      print(database, 'loaded')
  except FileNotFoundError:
    print('Unable to find', database)
    raise SystemExit(-1)

  # find number of parts for each type and compute upper bound
  nbods, nweaps, nwheels, ngads = [int(np.argwhere('nan' == db[p]['type'])[0]) for p in db]
  bound = nbods*(nweaps**max(db['body']['weapons']))*(nwheels**max(db['body']['wheels']))*(ngads**max(db['body']['gadgets']))

  # compute CATS configurations
  cats = np.full([bound, 1, 1, 1, 1, 1, 1], np.nan, dtype=(
    [('body', 'U6'), ('weapons', 'U18'), ('wheels', 'U21'), ('gadgets', 'U18'), ('health', 'f2'), ('damage', 'f2'), ('energy', 'f2')]))
  for bi, b in enumerate(db['body']):

    # compute combinations of parts
    weapons = it.combinations(range(nweaps), b['weapons'])
    wheels = it.combinations(range(nwheels), b['wheels'])
    gadgets = it.combinations(range(ngads), b['gadgets'])

    # compute attributes and index
    for wi, w in enumerate(weapons):
      for hi, h in enumerate(wheels):
        for gi, u in enumerate(gadgets):
          health = b['health'] + sum(db['wheels'][w, 'health']) + sum(db['gadgets'][u, 'health'])
          damage = sum(db['weapons'][w, 'damage'])
          energy = b['energy'] - sum(db['weapons'][w, 'energy']) - sum(db['gadgets'][u, 'energy'])
          idx = bi*np.prod(b['weapons', 'wheels', 'gadgets']) + wi*np.prod(b['wheels', 'gadgets']) + hi*b['gadgets'] + gi

          # store CATS configuration
          cats[idx] = [b['type'], ' '.join(w['type']), ' '.join(h['type']), ' '.join(u['type']), health, damage, energy]

  return scores


def bonus(parts):
  """
  Computes bonus attributes from parts
  Part modification relations:
  - body:   wheels, weapons
  - weapon: bodies
  - wheel:  bodies
  - gadget: bodies, weapons
  """

  return None

def load(plist='parts.txt'):
  """
  Helper function to write to the parts database from a text file
  """
  import csv

  try:
    with open(plist, 'r') as f:
      parts = list(csv.reader(f, skipinitialspace=True))
      print(plist, 'read')
  except FileNotFoundError:
    print('Unable to find', database)
    raise SystemExit(-1)

  # call write() to add to parts database
  write(parts)
  return 0


def write(parts, database='database.pkl', init_size=50):
  """
  Write to the parts database
  Field definitions:
  - body:   type, weapons, gadgets, wheels, health, energy, bonus
  - weapon: type, damage, energy, bonus
  - wheel:  type, health, bonus
  - gadget: type, health, energy, bonus
  """
  import numpy as np
  import pickle as pk

  # load (or create) the parts database
  fields = {'body':   [('type', 'U7'), ('weapons', 'f2'), ('gadgets', 'f2'), ('wheels', 'f2'), ('health', 'f2'), ('energy', 'f2'), ('bonus', 'U12'), ('modifier', 'f2')], 
            'weapon': [('type', 'U13'), ('damage', 'f2'), ('energy', 'f2'), ('bonus', 'U12'), ('modifier', 'f2')], 
            'wheel':  [('type', 'U7'), ('health', 'f2'), ('bonus', 'U12'), ('modifier', 'f2')], 
            'gadget': [('type', 'U9'), ('health', 'f2'), ('energy', 'f2'), ('bonus', 'U12'), ('modifier', 'f2')]}
  try:
    with open(database, 'rb') as f:
      db = pk.load(f)
      print(database, 'loaded')
  except FileNotFoundError:
    db = {p: np.full(init_size, np.nan, dtype=fields[p]) for p in fields}
    print(database, 'created')

  # add part(s) to the database
  if not isinstance(parts[0], list):
    parts = [parts]
  for p in parts:

    # find next free entry
    try:
      idx = int(np.argwhere('nan' == (db[p[0]]['type']))[0]))
    except ValueError:
      idx = 0
    
    # extend database if necessary
    if idx + 1 == db[p[0]].shape:
      db[p[0]] = np.concatenate((db[p[0]], np.full(init_size, np.nan, dtype=fields[p])))
      print(db, 'extended to', db[p[0]].shape)

    # padd part attributes if needed and write to the next available entry
    db[p[0]][idx] = tuple(np.pad(p[1:], (0, len(fields[p[0]]) - len(p[1:])), 'constant', constant_values=np.nan))
  print(len(parts), 'part(s) added')

  # save changes
  with open(database, 'wb') as f:
    pk.dump(db, f, pk.HIGHEST_PROTOCOL)
    print(database, 'saved')
  return 0


if __name__ == '__main__':
  """
  Create a parts database and return optimal CATS configurations
  """
  load()
  assemble()
  raise SystemExit(0)
