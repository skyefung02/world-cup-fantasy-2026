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
SCOUTING_BONUS_OWNERSHIP_PCT = 5   # the FIFA rule cutoff (percentSelected strictly below)
# We don't know deadline ownership, so eligibility is a soft ramp on *current* ownership
# rather than the hard <5% cliff: full bonus at/below LO, zero at/above HI, linear between.
# Tuned so a player sitting exactly at the 5% line keeps ~25% of the bonus (likely to cross,
# but not certain). LO controls how safe a genuine differential is; widen the band to soften.
SCOUTING_RAMP_LO = 3.5    # ownership % at/below which the bonus is applied in full
SCOUTING_RAMP_HI = 5.5    # ownership % at/above which the bonus is zeroed

# Set-piece / penalty allocation (model parameters, not FIFA scoring)
PEN_PROB = 0.20
PEN_CONVERSION = 0.78
SET_PIECE_ASSIST_PROB = 0.20