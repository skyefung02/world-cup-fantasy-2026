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
        "Edson Alvarez",       # CDM
        "Erik Lira",           # CM
        "Álvaro Fidalgo",      # CM
        "Roberto Alvarado",    # RW
        "Raul Jimenez",        # ST
        "Alexis Vega",         # LW
    ],

    "Czechia": [
        "Matej Kovar",         # GK
        "Štěpán Chaloupek",    # RB
        "Robin Hranáč",        # CB
        "Ladislav Krejci",     # CB
        "Vladimir Coufal",     # LB
        "Tomas Soucek",        # CDM
        "Vladimír Darida",     # CM
        "David Jurásek",       # CM
        "Lukáš Provod",        # RW
        "Pavel Šulc",          # ST
        "Patrik Schick",       # LW
    ],

    # ── Group B ────────────────────────────────────────────────────────────────
    "Canada": [
        "Maxime Crépeau",      # GK
        "Alistair Johnston",   # RB
        "Moïse Bombito",       # CB
        "Derek Cornelius",     # CB
        "Richie Laryea",       # LB
        "Tajon Buchanan",      # CDM
        "Ismael Kone",         # CM
        "Stephen Eustaquio",   # CM
        "Liam Millar",         # RW
        "Jonathan David",      # ST
        "Cyle Larin",          # LW
    ],

    "Bosnia and Herzegovina": [
        "Nikola Vasilj",       # GK
        "Amar Dedic",          # RB
        "Nikola Katic",        # CB
        "Tarik Muharemovic",   # CB
        "Sead Kolasinac",      # LB
        "Ivan Bašić",          # CDM
        "Benjamin Tahirovic",  # CM
        "Kerim Alajbegović'",  # CM
        "Amar Memić",          # RW
        "Ermedin Demirović",   # ST
        "Edin Dzeko",          # LW
    ],

    "Switzerland": [
        "Gregor Kobel",        # GK
        "Silvan Widmer",       # RB
        "Manuel Akanji",       # CB
        "Nico Elvedi",         # CB
        "Ricardo Rodriguez",   # LB
        "Granit Xhaka",        # CDM
        "Remo Freuler",        # CM
        "Michel Aebischer",    # CM
        "Dan Ndoye",           # RW
        "Breel Embolo",        # ST
        "Ruben Vargas",        # LW
    ],

    # ── Group C ────────────────────────────────────────────────────────────────
    "Brazil": [
        "Alisson",             # GK
        "Wesley",              # RB
        "Marquinhos",          # CB
        "Gabriel Magalhães",   # CB
        "Alex Sandro",         # LB
        "Bruno Guimaraes",     # CDM
        "Casemiro",            # CM
        "Matheus Cunha",       # CM
        "Raphinha",            # RW
        "Vinicius Júnior",     # ST
        "Luiz Henrique",       # LW
    ],

    "Morocco": [
        "Yassine Bounou",      # GK
        "Achraf Hakimi",       # RB
        "Nayef Aguerd",        # CB
        "Issa Diop",           # CB
        "Noussair Mazraoui",   # LB
        "Sofyan Amrabat",      # CDM
        "Neil El Aynaoui",     # CM
        "Azzedine Ounahi",     # CM
        "Brahim Diaz",         # RW
        "Ismael Saibari",      # ST
        "Abde Ezzalzouli",     # LW
    ],

    "Scotland": [
        "Angus Gunn",          # GK
        "Aaron Hickey",        # RB
        "Scott McKenna",       # CB
        "John Souttar",        # CB
        "Andrew Robertson",    # LB
        "Lewis Ferguson",      # CDM
        "Scott McTominay",     # CM
        "Ben Gannon-Doak",     # CM
        "John McGinn",         # RW
        "Ryan Christie",       # ST
        "Che Adams",           # LW
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
        "Timothy Weah",        # RW
        "Folarin Balogun",     # ST
        "Christian Pulisic",   # LW
    ],

    "Australia": [
        "Mathew Ryan",         # GK
        "Jordy Bos",           # RB
        "Harry Souttar",       # CB
        "Alessandro Circati",  # CB
        "Lucas Herrington",    # LB
        "Jacob Italiano",      # CDM
        "Jackson Irvine",      # CM
        "Aiden O'Neill",       # CM
        "Ajdin Hrustic",       # RW
        "Mo Toure",            # ST
        "Nestory Irankunda",   # LW
    ],

    "Türkiye": [
        "Uğurcan Çakır",       # GK
        "Zeki Çelik",          # RB
        "Merih Demiral",       # CB
        "Abdülkerim Bardakcı", # CB
        "Ferdi Kadioglu",      # LB
        "Hakan Çalhanoglu",    # CDM
        "Orkun Kokcu",         # CM
        "Barış Alper Yılmaz",  # CM
        "Arda Güler",          # RW
        "Kenan Yıldız",        # ST
        "Kerem Aktürkoğlu",    # LW
    ],

    # ── Group E ────────────────────────────────────────────────────────────────
    "Germany": [
        "Manuel Neuer",        # GK
        "Joshua Kimmich",      # RB
        "Jonathan Tah",        # CB
        "Nico Schlotterbeck",  # CB
        "David Raum",          # LB
        "Aleksandar Pavlovic", # CDM
        "Felix Nmecha",        # CM
        "Florian Wirtz",       # CM
        "Leroy Sane",          # RW
        "Jamal Musiala",       # ST
        "Kai Havertz",         # LW
    ],

    "Côte d'Ivoire": [
        "Yahia Fofana",        # GK
        "Wilfried Singo",      # RB
        "Ousmane Diomande",    # CB
        "Obite N'Dicka",       # CB
        "Ghislain Konan",      # LB
        "Franck Kessie",       # CDM
        "Ibrahim Sangare",     # CM
        "Seko Fofana",         # CM
        "Amad Diallo",         # RW
        "Yan Diomande",        # ST
        "Ange-Yoan Bonny",     # LW
    ],

    "Ecuador": [
        "Hernan Galindez",     # GK
        "Joel Ordóñez",        # RB
        "Willian Pacho",       # CB
        "Piero Hincapie",      # CB
        "Alan Franco",         # LB
        "Moises Caicedo",      # CDM
        "Pedro Vite",          # CM
        "Pervis Estupinan",    # CM
        "Gonzalo Plata",       # RW
        "John Yeboah",         # ST
        "Enner Valencia",      # LW
    ],

    # ── Group F ────────────────────────────────────────────────────────────────
    "Netherlands": [
        "Bart Verbruggen",     # GK
        "Denzel Dumfries",     # RB
        "Jurrien Timber",      # CB
        "Virgil Van Dijk",     # CB
        "Micky Van de Ven",    # LB
        "Ryan Gravenberch",    # CDM
        "Tijjani Reijnders",   # CM
        "Frenkie de Jong",     # CM
        "Donyell Malen",       # RW
        "Memphis",             # ST
        "Cody Gakpo",          # LW
    ],

    "Japan": [
        "Zion Suzuki",         # GK
        "Takehiro Tomiyasu",   # RB
        "Hiroki Ito",          # CB
        "Ko Itakura",          # CB
        "Ritsu Doan",          # LB
        "Kaishu Sano",         # CDM
        "Wataru Endo",         # CM
        "Keito Nakamura",      # CM
        "Takefusa Kubo",       # RW
        "Daichi Kamada",       # ST
        "Ayase Ueda",          # LW
    ],

    "Sweden": [
        "Kristoffer Nordfeldt", # GK
        "Isak Hien",           # RB
        "Carl Starfelt",       # CB
        "Victor Lindelof",     # CB
        "Gustaf Lagerbielke",  # LB
        "Yasin Ayari",         # CDM
        "Jesper Karlstrom",    # CM
        "Gabriel Gudmundsson", # CM
        "Anthony Elanga",      # RW
        "Viktor Gyokeres",     # ST
        "Alexander Isak",      # LW
    ],

    # ── Group G ────────────────────────────────────────────────────────────────
    "Belgium": [
        "Thibaut Courtois",    # GK
        "Thomas Meunier",      # RB
        "Zeno Debast",         # CB
        "Arthur Theate",       # CB
        "Maxim De Cuyper",     # LB
        "Amadou Onana",        # CDM
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
        "Fabián Ruiz",         # CM
        "Lamine Yamal",        # RW
        "Mikel Oyarzabal",     # ST
        "Nico Williams",       # LW
    ],

    "Uruguay": [
        "Sergio Rochet",       # GK
        "Guillermo Varela",    # RB
        "Ronald Araujo",       # CB
        "Sebastian Caceres",   # CB
        "Mathias Olivera",     # LB
        "Manuel Ugarte",       # CDM
        "Federico Valverde",   # CM
        "Rodrigo Bentancur",   # CM
        "Agustin Canobbio",    # RW
        "Maximiliano Araújo",  # ST
        "Darwin Nunez",        # LW
    ],
    
    # ── Group I ────────────────────────────────────────────────────────────────
    "France": [
        "Mike Maignan",        # GK
        "Jules Kounde",        # RB
        "William Saliba",      # CB
        "Dayot Upamecano",     # CB
        "Theo Hernandez",      # LB
        "Aurélien Tchouaméni", # CDM
        "Adrien Rabiot",       # CM
        "Desire Doue",         # CM
        "Michael Olise",       # RW
        "Kylian Mbappe",       # ST
        "Ousmane Dembele",     # LW
    ],
    
    "Senegal": [
        "Edouard Mendy",       # GK
        "Krepin Diatta",       # RB
        "Kalidou Koulibaly",   # CB
        "Moussa Niakhate",     # CB
        "El Hadji Malick Diouf", # LB
        "Idrissa Gana Gueye",  # CDM
        "Pape Gueye",          # CM
        "Iliman Ndiaye",       # CM
        "Habib Diarra",        # RW
        "Sadio Mane",          # ST
        "Nicolas Jackson",     # LW
    ],
    
    "Norway": [
        "Ørjan Nyland",        # GK
        "Julian Ryerson",      # RB
        "Kristoffer Ajer",     # CB
        "Torbjørn Heggem",     # CB
        "David Møller Wolfe",  # LB
        "Sander Berge",        # CDM
        "Fredrik Aursnes",     # CM
        "Martin Ødegaard",     # CM
        "Alexander Sørloth",   # RW
        "Erling Haaland",      # ST
        "Antonio Nusa",        # LW
    ],
    
    # ── Group J ────────────────────────────────────────────────────────────────
    "Austria": [
        "Alexander Schlager",  # GK
        "Konrad Laimer",       # RB
        "Philipp Lienhart",    # CB
        "David Alaba",         # CB
        "Stefan Posch",        # LB
        "Nicolas Seiwald",     # CDM
        "Xaver Schlager",      # CM
        "Patrick Wimmer",      # CM
        "Michael Gregoritsch", # RW
        "Marcel Sabitzer",     # ST
        "Marko Arnautovic",    # LW
    ],

    "Argentina": [
        "Emiliano Martinez",   # GK
        "Nahuel Molina",       # RB
        "Cristian Romero",     # CB
        "Nicolas Otamendi",    # CB
        "Nicolas Tagliafico",  # LB
        "Enzo Fernandez",      # CM
        "Alexis Mac Allister", # CM
        "Rodrigo de Paul",     # RW
        "Thiago Almada",
        "Lionel Messi",        # ST
        "Julian Alvarez",      # LW
    ],

    # ── Group K ────────────────────────────────────────────────────────────────
    "Portugal": [
        "Diogo Costa",         # GK
        "Joao Cancelo",        # RB
        "Ruben Dias",          # CB
        "Goncalo Inacio",      # CB
        "Nuno Mendes",         # LB
        "Vitinha",             # CDM
        "Joao Neves",          # CM
        "Bruno Fernandes",     # CM
        "Bernardo Silva",      # RW
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
        "Richard Rios",        # CM
        "Jhon Arias",          # CM
        "James Rodriguez",     # RW
        "Luis Diaz",           # ST
        "Luis Suarez",         # LW
    ],

    # ── Group L ────────────────────────────────────────────────────────────────
    "England": [
        "Jordan Pickford",     # GK
        "Reece James",         # RB
        "John Stones",         # CB
        "Marc Guehi",          # CB
        "Nico O'Reilly",       # LB
        "Declan Rice",         # CDM
        "Elliot Anderson",     # CM
        "Jude Bellingham",     # CM
        "Bukayo Saka",         # RW
        "Harry Kane",          # ST
        "Anthony Gordon",      # LW
    ],

    "Croatia": [
        "Dominik Livakovic",   # GK
        "Josip Stanisic",      # RB
        "Josip Sutalo",        # CB
        "Luka Vuskovic",       # CB
        "Josko Gvardiol",      # LB
        "Luka Modric",         # CDM
        "Mateo Kovacic",       # CM
        "Petar Sucic",         # CM
        "Andrej Kramaric",     # RW
        "Ante Budimir",        # ST
        "Ivan Perisic",        # LW
    ],
}
