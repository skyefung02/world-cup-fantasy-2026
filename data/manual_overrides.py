"""
Manual name overrides for World Cup fantasy → FBref player matching.

Keys   = WC player names as they appear in unmatched_players.csv
Values = FBref player names as they appear in the 'player' column of fbref_stats_reset.csv

Methodology
-----------
For each of the 580 unmatched WC players, top-3 FBref candidates (same nationality,
rapidfuzz token_sort_ratio) were examined. An override is included only when there is
genuine confidence the two entries refer to the same person. Common accepted cases:

  - Nickname / shortened first name  (Emi ↔ Emiliano, Nico ↔ Nicolás, Gio ↔ Giovanni,
                                       Maxi ↔ Maximiliano, Alex ↔ Alexander)
  - Single-name player whose full name appears in FBref  (Bremer, Ibañez, Trincão,
                                                           Cucho, Alisson Becker, Mousa Al Tamari)
  - Middle/extra name present in one source but not the other
    (Alisson Becker ↔ Alisson, Andrés Andrade ↔ Andrés Andrade Cedeño, etc.)
  - Transliteration variant  (Hawsawi ↔ Al Hosawi, Bu Washl ↔ Boushal, etc.)
  - Hyphenation / spacing difference  (Bonsu Baah ↔ Bonsu-Baah, Ati Zigi ↔ Ati-Zigi, etc.)
  - Goes by middle/second given name  (Obite Evan N'Dicka → listed as "Evan Ndicka" in WC)
  - Full nickname vs birth name  (Álex Grimaldo ↔ Alejandro Grimaldo)

Omitted if:
  - Fuzzy score was very low (< ~70) with no obvious linguistic link
  - First name clearly differs (José ≠ Juan, Hasan ≠ Hayder, Jordan ≠ André)
  - Player plays in a domestic league not covered by FBref (Qatar, Iraq domestic,
    Iran domestic, Jordan domestic, Uzbekistan domestic, etc.)
  - No FBref candidates exist for the nationality at all (Qatar)
  - Position mismatch with no compelling other evidence

Total overrides found: 33
"""

MANUAL_OVERRIDES = {

    # ── Argentina ──────────────────────────────────────────────────────────────
    # "Nico" is universally used; FBref full name "Nicolás Paz" (MF, Como)
    "Nico Paz": "Nicolás Paz",
    # "Emi" is the well-known nickname for Emiliano Buendía (MF, Aston Villa)
    "Emiliano Buendía": "Emi Buendía",

    # ── Australia ──────────────────────────────────────────────────────────────
    # "Jordan" is an alternate rendering of "Jordy" for Jordy Bos (DF, Holstein Kiel)
    "Jordan Bos": "Jordy Bos",

    # ── Brazil ─────────────────────────────────────────────────────────────────
    # Ibañez (Flamengo) = Roger Ibanez; single-name WC listing vs FBref full name
    "Ibañez": "Roger Ibanez",
    # Bremer (Juventus) = Gleison Bremer; single-name WC listing vs FBref full name
    "Bremer": "Gleison Bremer",
    # Alisson Becker is universally known; FBref lists him as "Alisson" (GK, Liverpool)
    "Alisson Becker": "Alisson",

    # ── Colombia ───────────────────────────────────────────────────────────────
    # "Cucho Hernández" → FBref "Cucho" (FW, Columbus Crew); Cucho is his established nickname
    "Cucho Hernández": "Cucho",

    # ── Côte d'Ivoire ──────────────────────────────────────────────────────────
    # Full birth name is "Obite Evan N'Dicka"; WC uses his professional name "Evan Ndicka"
    "Evan Ndicka": "Obite N'Dicka",

    # ── Egypt ──────────────────────────────────────────────────────────────────
    # "Nabil Emad Dunga" = "Nabil Dunga" (MF, Al Ahly) — middle name dropped in FBref
    "Nabil Emad Dunga": "Nabil Dunga",

    # ── Ghana ──────────────────────────────────────────────────────────────────
    # Spacing vs hyphen: "Christopher Bonsu Baah" = "Christopher Bonsu-Baah" (MF, Augsburg)
    "Christopher Bonsu Baah": "Christopher Bonsu-Baah",
    # Spacing vs hyphen: "Lawrence Ati Zigi" = "Lawrence Ati-Zigi" (GK, St Gallen)
    "Lawrence Ati Zigi": "Lawrence Ati-Zigi",
    # "Prince Adu" = "Prince Adu Kwabena" (FW,MF) — shortened first-name compound
    "Prince Adu": "Prince Adu Kwabena",

    # ── Haiti ──────────────────────────────────────────────────────────────────
    # Spacing vs hyphen: "Danley Jean Jacques" = "Danley Jean-Jacques" (MF, Strasbourg)
    "Danley Jean Jacques": "Danley Jean-Jacques",

    # ── Jordan ─────────────────────────────────────────────────────────────────
    # "Mousa Al Tamari" = "Musa Al-Taamari" (MF,FW, Montpellier) — transliteration variant
    "Mousa Al Tamari": "Musa Al-Taamari",

    # ── Morocco ────────────────────────────────────────────────────────────────
    # Hyphen vs space: "Ayoube Amaimouni-Echghouyab" = "Ayoube Amaimouni Echghouyab" (MF,FW)
    "Ayoube Amaimouni-Echghouyab": "Ayoube Amaimouni Echghouyab",

    # ── New Zealand ────────────────────────────────────────────────────────────
    # "Elijah Just" = "Eli Just" (MF,FW, FC Lausanne) — Elijah/Eli are the same
    "Elijah Just": "Eli Just",

    # ── Norway ─────────────────────────────────────────────────────────────────
    # "Leo Østigård" = "Leo Skiri Østigård" (DF, Lens) — middle name in FBref only
    "Leo Østigård": "Leo Skiri Østigård",

    # ── Panama ─────────────────────────────────────────────────────────────────
    # "Amir Murillo" = "Michael Amir Murillo" (DF,MF, Anderlecht) — goes by middle name
    "Amir Murillo": "Michael Amir Murillo",
    # "Andrés Andrade" = "Andrés Andrade Cedeño" (DF) — compound surname shortened in WC
    "Andrés Andrade": "Andrés Andrade Cedeño",
    # "Yoel Bárcenas" = "Édgar Yoel Bárcenas" (MF) — goes by middle name Yoel
    "Yoel Bárcenas": "Édgar Yoel Bárcenas",

    # ── Portugal ───────────────────────────────────────────────────────────────
    # "Trincão" (single name) = "Francisco Trincão" (MF, Sporting CP) — standard Portuguese nickname
    "Trincão": "Francisco Trincão",

    # ── Saudi Arabia ───────────────────────────────────────────────────────────
    # "Nawaf Bu Washl" = "Nawaf Boushal" (DF) — transliteration variant (same player)
    "Nawaf Bu Washl": "Nawaf Boushal",
    # "Zakaria Hawsawi" = "Zakaria Al Hosawi" (DF) — transliteration variant
    "Zakaria Hawsawi": "Zakaria Al Hosawi",
    # "Feras Al Brikan" = "Firas Al-Buraikan" (FW,MF) — transliteration variant
    "Feras Al Brikan": "Firas Al-Buraikan",

    # ── Senegal ────────────────────────────────────────────────────────────────
    # "Bara Sapoko Ndiaye" = "Bara Ndiaye" (MF) — middle name "Sapoko" dropped in FBref
    "Bara Sapoko Ndiaye": "Bara Ndiaye",

    # ── Spain ──────────────────────────────────────────────────────────────────
    # "Alejandro Grimaldo" = "Álex Grimaldo" (MF, Bayer Leverkusen) — goes by Álex professionally
    "Alejandro Grimaldo": "Álex Grimaldo",
    # "Fabián Ruiz" = "Fabián Ruiz Peña" (MF, PSG) — second surname present in FBref
    "Fabián Ruiz": "Fabián Ruiz Peña",

    # ── Mexico ─────────────────────────────────────────────────────────────────
    # "Jorge Sánchez" = "Jorge Eduardo Sánchez" (DF) — middle name in FBref
    "Jorge Sánchez": "Jorge Eduardo Sánchez",

    # ── Tunisia ────────────────────────────────────────────────────────────────
    # "Hadj Mahmoud" (MID) = "Mohamed Haj Mahmoud" (MF) — shortened/nickname version
    "Hadj Mahmoud": "Mohamed Haj Mahmoud",

    # ── Uruguay ────────────────────────────────────────────────────────────────
    # "Maxi Araújo" = "Maximiliano Araújo" (DF,MF, Atlético Madrid) — Maxi is standard nickname
    "Maxi Araújo": "Maximiliano Araújo",

    # ── Canada ─────────────────────────────────────────────────────────────────
    # "Ralph Priso" = "Ralph Priso-Mbongue" (MF,DF, Anderlecht) — hyphenated surname shortened
    "Ralph Priso": "Ralph Priso-Mbongue",

    # ── USA ────────────────────────────────────────────────────────────────────
    # "Alexander Freeman" = "Alex Freeman" (DF) — Alexander/Alex abbreviation
    "Alexander Freeman": "Alex Freeman",
    # "Giovanni Reyna" = "Gio Reyna" (MF, Nottm Forest) — Gio is his universally used name
    "Giovanni Reyna": "Gio Reyna",

    # ── Misc ────────────────────────────────────────────────────────────────────
    "Andy Robertson": "Andrew Robertson",
    "Juan Fernando Quintero": "Juan Quintero", 
    "Memphis Depay": "Memphis", 
    "Álex Zendejas": "Alejandro Zendejas",
    "Irfan Can Kahveci": "İrfan Kahveci",
    "Yéremy Pino": "Yeremi Pino",
    "Khojiakbar Alijonov": "Xojiakbar Alijonov",
    "Sherzod Nasrullaev": "Sherzod Nasrullayev", 
    "Abbosbek Fayzullaev": "Abbosbek Fayzullayev", 
    "Salem Al Dawsari": "Salem Al-Dawsari",
    "Nasser Al Dawsari": "Nasser Al-Dawsari",
    "Mohammed Al Owais": "Mohammed Al-Owais",
    "Raúl Rangel": "José Rangel Aguilar", 
    "Jens Castrop": "Castrop Jens",
    "Tino Livramento": "Valentino Livramento", 
    "Zizo": "Ahmed Sayed",
    "Mohammed Amoura": "Mohamed Amoura",
    "Yacine Titraoui": "Yassine Titraoui",
    "Nico González": "Nicolás González",
    "Cameron Devlin": "Cammy Devlin",
    "Álvaro Montero": "Álvaro David Montero",
    "Juan Portilla": "Juan Camilo Portilla",
    "Meschack Elia": "Meschak Elia",
    "Jean Michaël Seri": "Jean Seri",
    "Félix Torres": "Félix Torres Caicedo",
    "Hamdi Fathy": "Hamdy Fathy", 
    "Ahmed Fatouh": "Ahmed Aboul-Fotouh",
    "Hossam Abdelmaguid": "Hossam Abdelmegeed",
    "Mohamed El Shenawy": "Mohamed El-Shenawy",
    "Mostafa Shobeir": "Mostafa Shoubir", 
    "Abdul Baba": "Baba Rahman",
    "Abdul Fatawu": "Abdul Fatawu Issahaku",
    "Carl Sainté": "Carl Fred Sainté",
    "Mohammad Ghorbani": "Mohammed Ghorbani",
    "Hossein Kanani": "Hossein Kanaanizadegan",
    "Ehsan Hajisafi": "Ehsan Hajsafi",
    "Aria Yousefi": "Arya Yousefi",
    "Mohammad Mohebi": "Mohammad Mohebbi",
    "Mehdi Ghayedi": "Mehdi Ghaedi",
    "Frans Putros": "Frans Dhia Putros",
    "Munaf Younus": "Munaf Yunus",
    "Aymen Hussein": "Ayman Hussein",
    "Yazan Al Arab": "Yazan Al-Arab",
    "Husam Abu Al Dahab": "Husam Abu Dahab",
    "Mahmoud Al Mardi": "Mahmoud Al-Mardi",
    "Yazeed  Abulaila": "Yazeed Abulaila",
    "Abdallah Al Fakhouri": "Abdallah Al Fakhori",
    "Ehsan Haddad": "Ihsan Haddad",
    "Park Jin-Seob": "Park Jinseob",
    "Kim Jin-Gyu": "Kim Jin-kyu",
    "Cho Yu-Min": "Cho Yumin",
    "Kim Moon-Hwan": "Kim Moonhwan",
    "Lee Han-Beom": "Lee Hanbeom",
    "Lee Tae-Seok": "Lee Taeseok",
    "Kim Tae-Hyeon": "Kim Taehyeon",
    "Yang Hyun-Jun": "Yang Hyunjun",
    "Eom Ji-Sung": "Eom Jisung",
    "Song Bum-Keun": "Song Bumkeun",
    "Bae Jun-Ho": "Bae Junho",
    "Lee Gi-Hyuk": "Lee Gihyuk",
    "Munir El Kajoui": "Munir",
    "Matthew Garbett": "Matt Garbett",
    "Fredrik Bjørkan": "Fredrik André Bjørkan",
    "Éric Davis": "Erick Davis", 
    "Gustavo Velázquez": "Gustavo Velásquez",
    "Juan José Cáceres": "Juan Cáceres",
    "Mohammad Al Mannai": "Mohamed Manai", 
    "Bassam Al Rawi": "Bassam Al-Rawi",
    "Hashmi Al Hussain": "Al Hashmi Al Hussein", 
    "Hassan AlHaydos": "Hassan Al-Haydos",
    "Mahmud Abunada": "Mahmoud Abunada",
    "Sultan Al Brake": "Sultan Al-Brake",
    "Abdullah Al Khaibari": "Abdullah Al-Khaibari",
    "Abdulelah Al Amri": "Abdulelah Al-Amri", 
    "Hassan Kadish": "Hassan Kadesh",
    "Abdullah Al Hamddan": "Abdullah Al Hamdan",
    "Saleh Al Shehri": "Saleh Al-Shehri",
    "Sultan Mandash": "Sultan Mendash",
    "Ahmed Al Kassar": "Ahmed Al-Kassar",
    "Jehad Thikri": "Jehad Thakri",
    "Moteb Al Harbi": "Moteb Al-Harbi",
    "Mohammed Abu Al Shamat": "Mohammed Waheeb",
    "Saleh Abu Al Shamat": "Saleh Waheeb",
    "Abdullah Al Salem": "Abdulla Al Salem",
    "Yaya Sithole": "Sphephelo Sithole",
    "Adem Arous": "Adam Arous",
    "Abdulla Abdullaev": "Abdulla Abdullayev",
    "Aziz G'aniev": "Aziz G'aniyev",
    "Odiljon Hamrobekov": "Odiljon Xamrobekov",
    "Abbosbek Fayzullaev": "Abbosbek Fayzulllayev",
    "Azizbek Amanov": "Azizbek Amonov"
}



#The convention is - "FIFA NAME": "FBRef NAME",