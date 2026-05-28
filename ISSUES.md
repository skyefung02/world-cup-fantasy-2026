# Known Issues / To Revisit

## 1. League strength fallback for unmatched players

**Problem:** Unmatched players (those not found in the FBref scrape) currently receive `league_strength = 1.0`, the maximum value. This is wrong — most unmatched players are from domestic leagues not included in the scrape (e.g., Egyptian Premier League), which are weaker than the leagues we track.

The effect is that matched players from strong leagues (e.g., Salah) have their goal share diluted less than they should be, because unmatched teammates are assigned an inflated league strength.

**Proposed fix:** 
1. Add untracked domestic leagues to `league_strength.json` with appropriate strength values.
2. Build a `national team → representative domestic league` mapping in `build_projections.py`.
3. For unmatched players, fill `league_strength` from this mapping rather than defaulting to 1.0.

**When to action:** Once all World Cup squads are finalised. At that point, identify which national teams have a majority of players unmatched (i.e., playing in leagues not in the FBref scrape), and add those leagues to `league_strength.json`. The assumption is that unmatched players from a given national team play in that country's domestic top flight — this is a simplification but broadly correct.
