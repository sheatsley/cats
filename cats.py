"""
Ryan Sheatsley
6/14/2019
"""


def binomial_coeff(n, k):
  """
  Math is fun (Thanks Adrien and Eric)
  """
  import functools as ft
  import operator as op

  k = min(k, n - k)
  n = ft.reduce(op.mul, range(n, n - k, -1), 1)
  d = ft.reduce(op.mul, range(1, k + 1), 1)
  return n / d


def bonus(body, weapons, wheels, gadgets):
  """
  Computes bonus attributes from parts
  Part modification relations:
  - body:   weapons, wheels, gadgets
  - weapon: bodies, weapons(themselves), wheels, gadgets
  - wheel:  bodies
  - gadget: bodies, weapons
  """
  import numpy as np
  
  # check body bonuses
  health, damage = (0, 0)
  if body['bonus'] != 'nan':

    # body-to-weapon bonuses increase damage
    for w in weapons:
      if w['type'] == body['bonus']:
        damage += w['damage'] * body['modifier']

    # body-to-wheel/gadget bonuses increase health
    for hg in np.concatenate((wheels[['type', 'health']], gadgets[['type', 'health']])):
      if hg['type'] == body['bonus']:
        health += hg['health'] * body['modifier']

  # check weapon bonuses
  for w in weapons:
    if w['bonus'] != 'nan':
      part, attribute = w['bonus'].split('-')
      for bhg in np.concatenate(([body[['type', 'health']]], wheels[['type', 'health']], gadgets[['type', 'health']])):
        if bhg['type'] == part:

          # weapon-on-body bonuses increase damage
          if 'damage' == attribute: 
            damage += w['damage'] * w['modifier']

          # weapon-to-body/wheel/gadget bonuses increase health
          else:
            health += bhg['health'] * w['modifier']

  # check wheel bonuses
  for h in wheels:
    if h['bonus'] != 'nan':

      # wheel-to-body bonuses increase health
      if body['type'] == h['bonus']:
        health += body['health'] * h['modifier']

  # check gadget bonuses
  for g in gadgets:
    if g['bonus'] != 'nan':

      # gadget-to-weapon bonuses increase damage
      for w in weapons:
        if w['type'] == g['bonus']:
          damage += w['damage'] * g['modifier']

      # gadget-to-body bonuses increase health
      if body['type'] == g['bonus']:
        health += body['health'] * g['modifier']
  return health, damage 


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
  bound = int(sum([np.prod([sum([binomial_coeff(nweaps, k) for k in range(int(db['body'][i]['weapons']) + 1)]), 
    sum([binomial_coeff(nwheels, k) for k in range(int(db['body'][i]['wheels']) + 1)]), 
      sum([binomial_coeff(ngads, k) for k in range(int(db['body'][i]['gadgets']) + 1)])]) for i in range(nbods)]))

  # compute CATS configurations
  idx = 0
  cats = np.full(bound, np.nan, dtype=(
    [('body', 'U7'), ('weapons', 'U26'), ('wheels', 'U21'), ('gadgets', 'U18'), ('health', 'f2'), ('damage', 'f2'), ('energy', 'f2'), ('indicies', 'U28')]))
  for bi, b in enumerate(db['body'][:nbods]):

    # compute combinations of parts
    weapons = list(map(list, it.chain.from_iterable(it.combinations(range(nweaps), slots) for slots in range(int(b['weapons'])+1))))
    wheels = list(map(list, it.chain.from_iterable(it.combinations(range(nwheels), slots) for slots in range(int(b['wheels'])+1))))
    gadgets = list(map(list, it.chain.from_iterable(it.combinations(range(ngads), slots) for slots in range(int(b['gadgets'])+1))))

    # compute attributes and index
    for w in weapons:
      for h in wheels:
        for g in gadgets:
          health = b['health'] + np.sum(db['wheel'][h]['health']) + np.sum(db['gadget'][g]['health'])
          damage = np.sum(db['weapon'][w]['damage'])
          energy = b['energy'] - np.sum(db['weapon'][w]['energy']) - np.sum(db['gadget'][g]['energy'])

          # compute bonus attributes
          if b['bonus'] != 'nan' or (db['weapon'][w]['bonus'] != 'nan').any() or (db['wheel'][h]['bonus'] != 'nan').any() or (db['gadget'][g]['bonus'] != 'nan').any():
            bhealth, bdamage = bonus(b, db['weapon'][w], db['wheel'][h], db['gadget'][g])
            health += bhealth
            damage += bdamage

          # store CATS configuration
          cats[idx] = tuple([b['type'], ' '.join(db['weapon'][w]['type']), ' '.join(db['wheel'][h]['type']), ' '.join(db['gadget'][g]['type']), health, damage, energy,
            ''.join(str((bi, w, h, g)))])
          idx += 1
          print(str(int(idx / bound * 100))+'%', end='\r')
  return cats[:idx]


def score(cats, hweight=1., dweight=1., display=50, debug=True):
  """
  Computes scores for CATS vehicles and displays them
  CATS field layout:
  body, weapons,  wheels, gadgets, health, damage, energy
  """
  import code as cd
  import numpy as np

  # calculate the score for each CATS vehicle and sort
  scores = np.heaviside(cats['energy'], 1)*np.sum((hweight*cats['health'], dweight*cats['damage']), axis=0, dtype=int)
  best = np.argsort(scores)

  # determine max field length for pretty printing
  mbods, mweaps, mwheels, mgads = [len(str(max(cats[best[:-display:-1]][field], key=len))) for field in cats.dtype.names[:4]]
  mhp, mdmg, meng  = [len(str(max(cats[best[:-display:-1]][field].astype(int)))) for field in cats.dtype.names[4:7]]
  mscore = len(str(scores[best[-1]].astype(int)))
  fields = list(zip(cats.dtype.names, (mbods, mweaps, mwheels, mgads, mhp, mdmg, meng)))

  # print the top scores (index body weapons wheels gadgets health damage energy score)
  print('Crash Arena Turbo Stars'.center(mbods + mweaps + mwheels + mgads + mhp + mdmg + meng + mscore + 8))
  print('{:s}'.format('\u0332'.join(' '.join(('idx'[:len(str(display))], *[field[:flen].center(flen) for field, flen in fields], 'score'[:mscore].center(mscore))))))
  [print(str(idz+1).rjust(len(str(display))), 
    *[cats[idx][field].astype(str).ljust(flen) if idy < 4 else cats[idx][field].astype(int).astype(str).ljust(flen) for idy, (field, flen) in enumerate(fields)], 
    scores[idx].astype(int)) for idz, idx in enumerate(best[:-display:-1])]

  # drop to an interactive session if we're debugging
  if debug:
    code.interact(local=locals())
  return 0


def load(plist='parts.txt', comment='#'):
  """
  Helper function to write to the parts database from a text file
  """
  import csv
  import difflib
  import pickle as pk

  try:
    with open(plist, 'r') as f:
      parts = list(csv.reader(f, skipinitialspace=True))
      parts = [p for p in parts if not p[0].startswith(comment)]
      print(plist, 'read')
  except FileNotFoundError:
    print('Unable to find', database)
    raise SystemExit(-1)

  # compute deltas if this is not the first time
  try:
    with open('.'+plist, 'rb') as f:
      oparts = pk.load(f)
      loparts = [' '.join(p) for p in oparts]
      lparts = [' '.join(p) for p in parts]
      delta = list(difflib.ndiff(loparts, lparts))
      nparts = [(idx, p[1:].split()) for idx, p in enumerate(delta) if p[0] == '+']
      for idx, part in nparts:
        oparts.insert(idx, part)
      with open('.'+plist, 'wb') as f:
        pk.dump(oparts, f, pk.HIGHEST_PROTOCOL)
      parts = [part for idx, part in nparts]
  except FileNotFoundError:
    with open('.'+plist, 'wb') as f:
      pk.dump(parts, f, pk.HIGHEST_PROTOCOL)

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
  fields = {'body':   [('type', 'U13'), ('weapons', 'f2'), ('gadgets', 'f2'), ('wheels', 'f2'), ('health', 'f2'), ('energy', 'f2'), ('bonus', 'U13'), ('modifier', 'f2')], 
            'weapon': [('type', 'U13'), ('damage', 'f2'), ('energy', 'f2'), ('bonus', 'U13'), ('modifier', 'f2')], 
            'wheel':  [('type', 'U13'), ('health', 'f2'), ('bonus', 'U13'), ('modifier', 'f2')], 
            'gadget': [('type', 'U13'), ('health', 'f2'), ('energy', 'f2'), ('bonus', 'U13'), ('modifier', 'f2')]}
  try:
    with open(database, 'rb') as f:
      db = pk.load(f)
      print(database, 'loaded')
  except FileNotFoundError:
    db = {p: np.full(init_size, np.nan, dtype=fields[p]) for p in fields}
    print(database, 'created')

  # add part(s) to the database
  try:
    if not isinstance(parts[0], list):
      parts = [parts]
  except IndexError:
    pass
  for p in parts:

    # find next free entry
    try:
      idx = int(np.argwhere('nan' == db[p[0]]['type'])[0])
    except IndexError:
      idx = db[p[0]].shape[0]
    
    # extend database if necessary
    if idx == db[p[0]].shape[0]:
      db[p[0]] = np.concatenate((db[p[0]], np.full(init_size, np.nan, dtype=fields[p[0]])))
      print(database, 'extended to', db[p[0]].shape[0])

    # pad part attributes if needed and write to the next available entry
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
