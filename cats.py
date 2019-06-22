"""
Ryan Sheatsley
6/14/2019
"""


def binomial_coeff(n, k):
  """
  Math is fun (Thanks Adrien and Eric)
  """
  import math as mh
  return mh.factorial(n+k)/(mh.factorial(k)*mh.factorial(n))


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
  idx = 0
  cats = np.full(bound, np.nan, dtype=(
    [('body', 'U7'), ('weapons', 'U26'), ('wheels', 'U21'), ('gadgets', 'U18'), ('health', 'f2'), ('damage', 'f2'), ('energy', 'f2')]))
  for bi, b in enumerate(db['body'][:nbods]):

    # compute combinations of parts
    weapons = list(map(list, it.chain.from_iterable(it.combinations(range(nweaps), slots) for slots in range(int(b['weapons'])+1))))
    wheels = list(map(list, it.chain.from_iterable(it.combinations(range(nwheels), slots) for slots in range(int(b['wheels'])+1))))
    gadgets = list(map(list, it.chain.from_iterable(it.combinations(range(ngads), slots) for slots in range(int(b['gadgets'])+1))))

    # compute attributes and index
    for wi, w in enumerate(weapons):
      for hi, h in enumerate(wheels):
        for gi, u in enumerate(gadgets):
          health = b['health'] + np.sum(db['wheel'][h]['health']) + np.sum(db['gadget'][u]['health'])
          damage = np.sum(db['weapon'][w]['damage'])
          energy = b['energy'] - np.sum(db['weapon'][w]['energy']) - np.sum(db['gadget'][u]['energy'])
          #TODO incorporate bonus() here

          # store CATS configuration
          cats[idx] = tuple([b['type'], ' '.join(db['weapon'][w]['type']), ' '.join(db['wheel'][h]['type']), ' '.join(db['gadget'][u]['type']), health, damage, energy])
          idx += 1
  return cats[:idx]


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


def score(cats, hweight=1., dweight=1., display=50):
  """
  Computes scores for CATS vehicles and displays them
  CATS field layout:
  body, weapons,  wheels, gadgets, health, damage, energy
  """
  import numpy as np

  # calculate the score for each CATS vehicle and sort
  scores = np.heaviside(cats['energy'], 1)*np.sum((hweight*cats['health'], dweight*cats['damage']), axis=0, dtype=int)
  best = np.argsort(scores)

  # determine max field length for pretty printing
  mbods, mweaps, mwheels, mgads = [len(str(max(cats[best[:-display:-1]][field], key=len))) for field in cats.dtype.names[:4]]
  mhp, mdmg, meng  = [len(str(max(cats[best[:-display:-1]][field].astype(int)))) for field in cats.dtype.names[4:]]
  mscore = len(str(scores[best[-1]].astype(int)))
  fields = list(zip(cats.dtype.names, (mbods, mweaps, mwheels, mgads, mhp, mdmg, meng)))

  # print the top scores (index body weapons wheels gadgets health damage energy score)
  print('Crash Arena Turbo Stars'.center(mbods + mweaps + mwheels + mgads + mhp + mdmg + meng + mscore + 8))
  print('{:s}'.format('\u0332'.join(' '.join(('idx'[:len(str(display))], *[field[:flen].center(flen) for field, flen in fields], 'score'[:mscore].center(mscore))))))
  [print(str(idz+1).rjust(len(str(display))), 
    *[cats[idx][field].astype(str).ljust(flen) if idy < 4 else cats[idx][field].astype(int).astype(str).ljust(flen) for idy, (field, flen) in enumerate(fields)], 
    scores[idx].astype(int)) for idz, idx in enumerate(best[:-display:-1])]
  return 0


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
  score(assemble())
  raise SystemExit(0)
