"""
Manual starting XI overrides per team. Used in 05_international_data_fbref.ipynb
to bypass the algorithmic top-10-by-mp_share starter selection for teams where
we have high confidence in the expected matchday XI.

Keys   = team names as they appear in wc_roster["team"]
Values = list of 11 player names (using roster spellings; matched via to_ascii
         after MANUAL_OVERRIDES substitution).

Teams not in this dict fall back to the algorithmic starter selection.
"""

MANUAL_STARTING_XI = {

    # ── Group A ────────────────────────────────────────────────────────────────
    "Mexico": [
        "Raúl Rangel",         # GK
        "Israel Reyes",        # RB
        "Cesar Montes",        # CB
        "Johan Vasquez",       # CB
        "Jesus Gallardo",      # LB
        "Luis Romo",       # CDM
        "Erik Lira",           # CM
        "Gilberto Mora",      # CM
        "Roberto Alvarado",    # RW
        "Raul Jimenez",        # ST
        "Julian Quinones",         # LW
    ],
    
    # ── Group B ────────────────────────────────────────────────────────────────
    "Canada": [
        "Maxime Crépeau",      # GK
        "Alistair Johnston",   # RB
        "Moise Bombito",       # CB
        "Derek Cornelius",     # CB
        "Richie Laryea",       # LB
        "Tajon Buchanan",      # CDM
        "Nathan Saliba",         # CM
        "Stephen Eustaquio",   # CM
        "Liam Millar",         # RW
        "Jonathan David",      # ST
        "Cyle Larin",          # LW
    ],

    "Switzerland": [
        "Gregor Kobel",        # GK
        "Denis Zakaria",       # RB
        "Manuel Akanji",       # CB
        "Nico Elvedi",         # CB
        "Ricardo Rodriguez",   # LB
        "Granit Xhaka",        # CDM
        "Remo Freuler",        # CM
        "Johan Manzambi",    # CM
        "Dan Ndoye",           # RW
        "Breel Embolo",        # ST
        "Ruben Vargas",        # LW
    ],

    # ── Group C ────────────────────────────────────────────────────────────────
    "Brazil": [
        "Alisson",             # GK
        "Danilo",              # RB
        "Marquinhos",          # CB
        "Gabriel Magalhães",   # CB
        "Douglas Santos",         # LB
        "Bruno Guimaraes",     # CDM
        "Casemiro",            # CM
        "Lucas Paqueta",       # CM
        "Rayan",            # RW
        "Vinicius Júnior",     # ST
        "Matheus Cunha",       # LW
    ],

    "Morocco": [
        "Yassine Bounou",      # GK
        "Achraf Hakimi",       # RB
        "Chadi Riad",        # CB
        "Issa Diop",           # CB
        "Noussair Mazraoui",   # LB
        "Ayyoub Bouaddi",      # CDM
        "Neil El Aynaoui",     # CM
        "Azzedine Ounahi",     # CM
        "Brahim Diaz",         # RW
        "Ismael Saibari",      # ST
        "Bilal El Khannouss",     # LW
    ],

    # ── Group D ────────────────────────────────────────────────────────────────
    "USA": [
        "Matt Freese",         # GK
        "Alexander Freeman",   # RB
        "Chris Richards",      # CB
        "Tim Ream",            # CB
        "Antonee Robinson",    # LB
        "Tyler Adams",         # CDM
        "Weston McKennie",     # CM
        "Malik Tillman",       # CM
        "Sergino Dest",        # RW
        "Ricardo Pepi",     # ST
        "Christian Pulisic",   # LW
    ],

    # ── Group E ────────────────────────────────────────────────────────────────
    # ── Group F ────────────────────────────────────────────────────────────────
    # ── Group G ────────────────────────────────────────────────────────────────
    "Belgium": [
        "Thibaut Courtois",    # GK
        "Timothy Castagne",      # RB
        "Arthur Theate",         # CB
        "Brandon Mechele",       # CB
        "Maxim De Cuyper",     # LB
        "Hans Vanaken",        # CDM
        "Youri Tielemans",     # CM
        "Kevin de Bruyne",     # CM
        "Charles de Ketelaere", # RW
        "Leandro Trossard",    # ST
        "Jeremy Doku",         # LW
    ],

    # ── Group H ────────────────────────────────────────────────────────────────
    "Spain": [
        "Unai Simon",          # GK
        "Marcos Llorente",     # RB
        "Pau Cubarsi",         # CB
        "Aymeric Laporte",     # CB
        "Marc Cucurella",      # LB
        "Rodri",               # CDM
        "Pedri",               # CM
        "Dani Olmo",         # CM
        "Lamine Yamal",        # RW
        "Mikel Oyarzabal",     # ST
        "Alex Baena",       # LW
    ],

    # ── Group I ────────────────────────────────────────────────────────────────
    "France": [
        "Mike Maignan",        # GK
        "Jules Kounde",        # RB
        "William Saliba",      # CB
        "Dayot Upamecano",     # CB
        "Lucas Digne",      # LB
        "Manu Kone", # CDM
        "Adrien Rabiot",       # CM
        "Bradley Barcola",         # CM
        "Michael Olise",       # RW
        "Kylian Mbappe",       # ST
        "Ousmane Dembele",     # LW
    ],
  
    "Norway": [
        "Ørjan Nyland",        # GK
        "Marcus Pedersen",      # RB
        "Kristoffer Ajer",     # CB
        "Torbjørn Heggem",     # CB
        "David Møller Wolfe",  # LB
        "Sander Berge",        # CDM
        "Patrick Berg",     # CM
        "Martin Ødegaard",     # CM
        "Alexander Sørloth",   # RW
        "Erling Haaland",      # ST
        "Antonio Nusa",        # LW
    ],
    
    # ── Group J ────────────────────────────────────────────────────────────────
    "Argentina": [
        "Emiliano Martinez",   # GK
        "Nahuel Molina",       # RB
        "Cristian Romero",     # CB
        "Lisandro Martinez",    # CB
        "Facundo Medina",  # LB
        "Enzo Fernandez",      # CM
        "Alexis Mac Allister", # CM
        "Rodrigo de Paul",     # RW
        "Thiago Almada",
        "Lionel Messi",        # ST
        "Lautaro Martinez",      # LW
    ],

    # ── Group K ────────────────────────────────────────────────────────────────
    "Portugal": [
        "Diogo Costa",         # GK
        "Joao Cancelo",        # RB
        "Ruben Dias",          # CB
        "Renato Veiga",      # CB
        "Nuno Mendes",         # LB
        "Vitinha",             # CDM
        "Joao Neves",          # CM
        "Bruno Fernandes",     # CM
        "Rafael Leao",      # RW
        "Pedro Neto",          # ST
        "Cristiano Ronaldo",   # LW
    ],

    "Colombia": [
        "Camilo Vargas",       # GK
        "Daniel Munoz",        # RB
        "Davinson Sanchez",    # CB
        "Jhon Lucumi",         # CB
        "Johan Mojica",        # LB
        "Jefferson Lerma",     # CDM
        "Gustavo Puerta",        # CM
        "Jhon Arias",          # CM
        "James Rodriguez",     # RW
        "Luis Diaz",           # ST
        "Luis Suarez",         # LW
    ],

    # ── Group L ────────────────────────────────────────────────────────────────
    "England": [
        "Jordan Pickford",     # GK
        "Djed Spence",         # RB
        "Marc Guehi",         # CB
        "Ezri Konsa",          # CB
        "Nico O'Reilly",       # LB
        "Declan Rice",         # CDM
        "Elliot Anderson",     # CM
        "Jude Bellingham",     # CM
        "Noni Madueke",         # RW
        "Harry Kane",          # ST
        "Anthony Gordon",      # LW
    ],
}
