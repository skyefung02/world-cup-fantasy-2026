# scoring.py
# FIFA World Cup 2026 Fantasy scoring system (static, won't change)

# Appearance points (all positions)
APPEARANCE_SUB = 1       # Up to 60 minutes
APPEARANCE_FULL = 2      # 60+ minutes (1 + 1)

# All players
ASSIST_PTS = 3
YELLOW_CARD_PTS = -1
RED_CARD_PTS = -2
OWN_GOAL_PTS = -2
WIN_PENALTY_PTS = 2
CONCEDE_PENALTY_PTS = -1

# Goals scored by position
GOAL_PTS = {
    "GK":  9,
    "DEF": 7,
    "MID": 6,
    "FWD": 5,
}

# Clean sheet by position (must play 60+ mins)
CLEAN_SHEET_PTS = {
    "GK":  5,
    "DEF": 5,
    "MID": 1,
    "FWD": 0,
}

# Goals conceded (GK and DEF only)
# First goal conceded: 0 pts, each additional: -1
GOALS_CONCEDED_PTS = {
    "GK":  -1,   # per goal after first
    "DEF": -1,   # per goal after first
    "MID":  0,
    "FWD":  0,
}

# Position-specific bonuses
SAVES_PTS = 1        # GK: every 3 saves
TACKLES_PTS = 1      # MID: every 3 tackles
CHANCES_CREATED_PTS = 1   # MID: every 2 chances created
SHOTS_ON_TARGET_PTS = 1   # FWD: every 2 shots on target

# Bonus points
FREE_KICK_GOAL_PTS = 1    # in addition to goal points
SCOUTING_BONUS_PTS = 2    # >4pts in match + <5% ownership
SCOUTING_BONUS_OWNERSHIP_PCT = 5   # eligibility cutoff (percentSelected strictly below)

# Set-piece / penalty allocation (model parameters, not FIFA scoring)
PEN_PROB = 0.20
PEN_CONVERSION = 0.78
SET_PIECE_ASSIST_PROB = 0.20