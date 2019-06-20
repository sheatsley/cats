"""
Ryan Sheatsley
6/14/2019
"""


def assemble(parts, database='database.pkl'):
  """
  Exhaustively compute CATS combinations from the parts database
  Field definitions:
  - body:    type, weapons, utilities, wheels, health, energy, bonus
  - weapon:  type, damage, energy, bonus
  - wheel:   type, health, bonus
  - utility: type, health, energy, bonus
  """
  import numpy as np
  import itertools as it

  # load the parts databse
  try:
    with open(database, 'rb') as f:
      db = pickle.load(f)
      print(database, 'loaded')
  except FileNotFoundError:
    print('Unable to find', database)
    raise SystemExit(-1)

  # find number of parts for each type
  nbods, nweaps, nwheels, nutils = [min(np.argwhere(np.isnan(db[p].T[0]))) for p in db]

  # compute CATS configurations
  cats = np.full([nbods, 1, 1, 1, 1, 1, 1], np.nan, dtype=(
    [('body', 'S6'), ('weapons', 'S18'), ('wheels', 'S21'), ('utilities', 'S18'), ('health', 'S21'), ('damage', 'S21'), ('energy', 'S21')]))
  for b in db['body']:

    # compute combinations of parts
    weapons = it.combinations(range(nweaps), b['weapons'])
    wheels = it.combinations(range(nwheels), b['wheels'])
    utilities = it.combinations(range(nutils), b['utilities'])

    # compute metrics
    for  in weapons:
      if db[



  return scores


def write(parts, database='database.pkl', init_size=50):
  """
  Write to the parts database
  Field definitions:
  - body:    type, weapons, utilities, wheels, health, energy, bonus
  - weapon:  type, damage, energy, bonus
  - wheel:   type, health, bonus
  - utility: type, health, energy, bonus
  """
  import numpy as np

  # load (or create) the parts database
  try:
    with open(database, 'rb') as f:
      db = pickle.load(f)
      print(database, 'loaded')
  except FileNotFoundError:
    fields = {'body':    [('type', 'S6'), ('weapons', 'f1'), ('utilities', 'f2'), ('wheels', 'f2'), ('health', 'f2'), ('energy', 'f2'), ('bonus', 'f2')], 
              'weapon':  [('type', 'S6'), ('damage', 'f2'), ('energy', 'f2'), ('bonus', 'f2')], 
              'wheel':   [('type', 'S7'), ('health', 'f2'), ('bonus', 'f2')], 
              'utility': [('type', 'S6'), ('health', 'f2'), ('energy', 'f2'), ('bonus', 'f2')]}
    db = {p: np.full([init_size, len(fields[p])], np.nan, dtype=fields[p]) for p in fields}
    print(db, 'created')

  # add part(s) to the database
  if not isistance(parts[0], list):
    parts = [parts]
  for p in parts:
    idx = min(np.argwhere(np.isnan(db[p[0]].T[0])))
    if idx + 1 == db[pt[0]].T[0].shape:
      db[p[0]] = np.concatenate((db[p[0]], np.full([init_size, len(fields[p[0]])], np.nan)))
      print(db, 'extended to', db[p[0]].shape)
  print(len(parts), 'part(s) added')

  # save changes
  with open(database, 'wb') as f:
    pickle.dump(db, f, pickle.HIGHEST_PROTOCOL)
    print(db, 'saved')
  return
