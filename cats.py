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
  bound = int(sum([np.prod([binomial_coeff(n, k) for n,k in zip([nweaps, nwheels, ngads], db['body'][i][['weapons', 'wheels', 'gadgets']])]) for i in range(nbods)]))

  # compute CATS configurations
  cats = np.full(bound, np.nan, dtype=(
    [('body', 'U7'), ('weapons', 'U26'), ('wheels', 'U21'), ('gadgets', 'U18'), ('health', 'f2'), ('damage', 'f2'), ('energy', 'f2')]))
  for bi, b in enumerate(db['body'][:nbods]):

    # compute combinations of parts
    weapons = map(list, it.chain.from_iterable(it.combinations(range(nweaps), slots) for slots in range(int(b['weapons'])+1)))
    wheels = map(list, it.chain.from_iterable(it.combinations(range(nwheels), slots) for slots in range(int(b['wheels'])+1)))
    gadgets = map(list, it.chain.from_iterable(it.combinations(range(ngads), slots) for slots in range(int(b['gadgets'])+1)))

    # compute attributes and index
    for wi, w in enumerate(weapons):
      for hi, h in enumerate(wheels):
        for gi, u in enumerate(gadgets):
          health = b['health'] + np.sum(db['wheel'][h]['health']) + np.sum(db['gadget'][u]['health'])
          damage = np.sum(db['weapon'][w]['damage'])
          energy = b['energy'] - np.sum(db['weapon'][w]['energy']) - np.sum(db['gadget'][u]['energy'])
          idx = int(bi*np.prod(b[['weapons', 'wheels', 'gadgets']].astype(list)) + wi*np.prod(b[['wheels', 'gadgets']].astype(list)) + hi*b['gadgets'] + gi)

          # store CATS configuration
          cats[idx] = tuple([b['type'], ' '.join(db['weapon'][w]['type']), ' '.join(db['wheel'][h]['type']), ' '.join(db['gadget'][u]['type']), health, damage, energy])

  # show 
  print('weewoo')
  pass

  return scores


def binomial_coeff(n, k):
  """
  Math is fun (Thanks Adrien and Eric)
  """
  import math as mh
  return mh.factorial(n+k)/(mh.factorial(k)*mh.factorial(n))


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
      idx = int(np.argwhere('nan' == (db[p[0]]['type']))[0])
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
