## CATS 

This is *CATS* - a repo for computing optimal builds for [Crash Arena Turbo
Stars](https://www.catsthegame.com). C.A.T.S. is a PvP battle bots game where
players design vehicles and take them head-to-head against other
player-designed machines. Vehicles are comprised from a series of different
parts that affect the overall attributes of the machine. Each part has their
own pros and cons; this repo aids players in informing part configurations
that are most optimal from what they currently have (it even incorporates
[Magic Bonuses](https://catsthegame.fandom.com/wiki/Magic_Bonus) when computing
optimal builds). Under the hood, finding an optimal build is reduced to solving
a [knapsack-like problem](https://en.wikipedia.org/wiki/Knapsack_problem). 

## How to use this repo

Requirements:
- Python 3
- numpy (`pip3 install numpy`)

To start off, run `python3 cats.py` to confirm that the required python
modules are available. Depending on your machine, it can take some time (and
will give you a sense of computational complexity for a moderate number of
parts). If everything is successfull, you should see the following output:
`
$ python3 cats.py
parts.txt read
database.pkl loaded
0 part(s) added
database.pkl saved
database.pkl loaded
                       Crash Arena Turbo Stars
i̲d̲ ̲ ̲ ̲b̲o̲d̲y̲ ̲ ̲ ̲ ̲ ̲ ̲w̲e̲a̲p̲o̲n̲s̲ ̲ ̲ ̲ ̲ ̲ ̲ ̲ ̲ ̲ ̲ ̲ ̲ ̲w̲h̲e̲e̲l̲s̲ ̲ ̲ ̲ ̲ ̲ ̲ ̲ ̲g̲a̲d̲g̲e̲t̲s̲ ̲ ̲h̲e̲a̲ ̲d̲a̲m̲ ̲e̲ ̲s̲c̲o
 1 pyramid minigun         sticky roller roller  forklift 245 210 1 455
 2 pyramid minigun         sticky roller roller  harpoon  245 210 2 455
 3 pyramid minigun         sticky roller knob    harpoon  235 210 2 445
 4 pyramid minigun         sticky roller roller  harpoon  235 210 2 445
`

### Adding parts

The script starts with `parts.txt` (an example is provided to demonstrate how
to describe parts correctly).  `parts.txt` constitutes a simple comma-separated
list of parts to be considered for computing optimal builds (to help organize
the parts list, comments are supported; any line leading with `#` are ignored).
Parts are parsed as follows:

- bodies: type, weapon slots, gadget slots, wheel slots, health, energy
  capacity, bonus type, bonus modifier
- weapons: type, damage, energy cost, bonus type, bonus modifier
- wheels: type, health, bonus type, bonus modifier
- gadgets: type, health, enegy cost, bonus type, bonus modifier

For example, `body, pyramid, 2, 1, 2, 120, 13, minigun, 1.00` describes a
pyramid body, with 2 weapon slots, 1 gadget slot, 2 wheels, 120 base health,
and a 100% bonus modifier for miniguns. At this time, [Ultimate
Machines](https://catsthegame.fandom.com/wiki/Ultimate_Machines) are not
properly supported.

You can routinely add parts to `parts.txt` as you obtain them. Note that,
in the worst case, the runtime is roughly O(&micro;%sum;(&theta;!)), where
&micro; is the number of bodies and &theta; represents the number of weapons,
wheels, or gadgets. Therefore, regularly scrapping lower-grade parts is highly
recommended.

### Emphasizing specific configurations

The "score" of a configuration is a simple weighted sum, given by: `score =
&theta;<sub>h</sub>*h + &theta;<sub>d</sub>*d` where `h` is the total health,
`d` is the total weapon damage, `&theta;<sub>h/d</sub>` are the weights
associated with health and damage, respectively. By default, they are set to
1.0. You can, for example, emphasize damage-heavy configurations by increasing
`dweight` in the declaration of `score` on line 243 of `cats.py`.

### Interpreting the results

After the scores are computed, a list of configurations is shown (the number of
configurations is controlled by `display` in the declaration of `score`),
sorted by decreasing score. The columns are described as follows:

- id: relative position of the configuration (with respect to score)
- body: body type used
- weapons: weapons used
- wheels: wheels used
- gadgets: gadgets used
- health: total health of the vehicle
- damage: total damage of the vehicle
- energy: leftover energy
- score: configuration score

At this time, the _exact_ parts used is not described (i.e., if you have
multiple minigun weapons, which mini-gun in your parts list is not shown.
However, they should be relatively inferable. For example, the optimal build in
the provided parts list uses a pyramid body with 1 minigun. Within the
parts list, there is one pyramid body that has a 100% magic bonus for
miniguns, so those are likely the parts in the optimal build. To confirm,
simply assemble the parts within C.A.T.S. and make sure the reported health and
damage match those shown in the display.
