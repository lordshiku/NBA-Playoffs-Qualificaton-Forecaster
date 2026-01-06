# ---- Libraries ----
library(dplyr)
library(readr)

#important note, used claude for formatting and readability of code, plus adding commments

setwd("removed for author safety")

# ---- Output paths (ONLY the 3 requested CSVs: nominal, no Season) ----
path_out_train_nom_noSeason <- "removed for author safety"
path_out_dev_nom_noSeason   <- "removed for author safety"
path_out_test_nom_noSeason  <- "removed for author safety"

# ---- Load raw game-level data (pick file interactively) ----
# Expected columns:
# RegularSeason, Season, Date, HomeTeam, HomeTeamScore, AwayTeam, AwayTeamScore,
# HomeTeamWin, AwayTeamWin, NeutralTeamWin (NeutralTeamWin optional)
games <- read.csv(file.choose(), stringsAsFactors = FALSE)

# ---- Validate input ----
if (!"RegularSeason" %in% names(games)) {
  stop("Column 'RegularSeason' not found in the input file.")
}
required_cols <- c("Season","Date","HomeTeam","HomeTeamScore","AwayTeam","AwayTeamScore",
                   "HomeTeamWin","AwayTeamWin")
missing_cols <- setdiff(required_cols, names(games))
if (length(missing_cols) > 0) {
  stop(sprintf("Input file missing required columns: %s", paste(missing_cols, collapse = ", ")))
}

# ---- Identify playoff teams from NON-regular-season games ----
playoffs_raw <- games %>%
  filter(!(RegularSeason %in% c(1, "1", TRUE, "TRUE", "true")))

playoff_teams <- bind_rows(
  playoffs_raw %>% transmute(Season, Team = HomeTeam),
  playoffs_raw %>% transmute(Season, Team = AwayTeam)
) %>%
  distinct() %>%
  mutate(madePlayoffs = 1L)

# ---- Now restrict to REGULAR-season games for feature computation ----
games_reg <- games %>%
  filter(RegularSeason %in% c(1, "1", TRUE, "TRUE", "true"))

# ---- Parse dates (best-effort) to order games; fallback to row order ----
parse_maybe <- function(x) {
  suppressWarnings({
    d <- as.Date(x)
    if (all(is.na(d))) d <- as.Date(x, format = "%m/%d/%Y")
    if (all(is.na(d))) d <- as.Date(x, format = "%Y-%m-%d")
    d
  })
}
games_reg <- games_reg %>%
  mutate(Date_parsed = parse_maybe(Date),
         RowOrder = row_number())  # stable fallback

# ---- Normalize to team-game long format (one row per team per REGULAR-SEASON game) ----
home_rows <- games_reg %>%
  transmute(
    Season,
    Date_parsed,
    RowOrder,
    Team          = HomeTeam,
    Opp           = AwayTeam,
    PointsFor     = HomeTeamScore,
    PointsAgainst = AwayTeamScore,
    PointDiff     = HomeTeamScore - AwayTeamScore,
    Win           = as.integer(HomeTeamWin == 1),
    Home          = 1L
  )

away_rows <- games_reg %>%
  transmute(
    Season,
    Date_parsed,
    RowOrder,
    Team          = AwayTeam,
    Opp           = HomeTeam,
    PointsFor     = AwayTeamScore,
    PointsAgainst = HomeTeamScore,
    PointDiff     = AwayTeamScore - HomeTeamScore,
    Win           = as.integer(AwayTeamWin == 1),
    Home          = 0L
  )

long <- bind_rows(home_rows, away_rows)

# ===============================================================================
# NEW FEATURE: SCHEDULE STRENGTH - AvgOppWinPct
# ===============================================================================
# CRITICAL TEMPORAL CONSTRAINT:
# We must compute opponent strength using ONLY information available at the time
# of prediction (first 40% of season). This prevents data leakage.
# 
# Algorithm:
# 1. For each team-season, identify their first 40% of games (chronologically)
# 2. For each opponent faced, compute that opponent's WinPct using ONLY their 
#    first 40% of games
# 3. Average these opponent WinPcts to get schedule strength metric
#
# This ensures we're not using future information (opponent's full-season record)
# to predict early-season outcomes.
# ===============================================================================

# Step 1: Compute early-season WinPct for ALL teams (to serve as opponent strength lookup)
# This creates a lookup table: (Season, Team) -> EarlyWinPct
early_winpct_lookup <- long %>%
  group_by(Season, Team) %>%
  arrange(Date_parsed, RowOrder, .by_group = TRUE) %>%
  mutate(TotalGames = n(),
         TakeN = ceiling(TotalGames * 0.40),
         GameIndex = row_number()) %>%
  filter(GameIndex <= TakeN) %>%
  summarise(
    EarlyWinPct = mean(Win, na.rm = TRUE),
    .groups = "drop"
  )

# Step 2: Keep only the FIRST 40% of each team's regular-season games ----
long_first_40 <- long %>%
  group_by(Season, Team) %>%
  arrange(Date_parsed, RowOrder, .by_group = TRUE) %>%
  mutate(TotalGames = n(),
         TakeN = ceiling(TotalGames * 0.40),
         GameIndex = row_number()) %>%
  filter(GameIndex <= TakeN) %>%
  ungroup()

# Step 3: Attach opponent's early-season WinPct to each game in the first 40%
long_first_40_with_opp <- long_first_40 %>%
  left_join(
    early_winpct_lookup,
    by = c("Season" = "Season", "Opp" = "Team")
  ) %>%
  rename(OppEarlyWinPct = EarlyWinPct)

# Step 4: Compute AvgOppWinPct for each team-season (average opponent strength)
schedule_strength <- long_first_40_with_opp %>%
  group_by(Season, Team) %>%
  summarise(
    AvgOppWinPct = mean(OppEarlyWinPct, na.rm = TRUE),
    .groups = "drop"
  )

# ---- Team-season feature aggregation from FIRST 40% only (ORIGINAL FEATURES) ----
team_season_early <- long_first_40 %>%
  group_by(Season, Team) %>%
  summarise(
    GamesPlayedEarly   = n(),
    GamesWonEarly      = sum(Win, na.rm = TRUE),
    WinPct             = if_else(GamesPlayedEarly > 0, GamesWonEarly / GamesPlayedEarly, NA_real_),
    
    HomeGamesEarly     = sum(Home == 1L, na.rm = TRUE),
    HomeWinsEarly      = sum(Win * (Home == 1L), na.rm = TRUE),
    HomeWinPct         = if_else(HomeGamesEarly > 0, HomeWinsEarly / HomeGamesEarly, NA_real_),
    
    BlowoutWinsEarly   = sum((Win == 1L) & (PointDiff >= 15), na.rm = TRUE),
    BlowoutWinPct      = if_else(GamesWonEarly > 0, BlowoutWinsEarly / GamesWonEarly, NA_real_),
    
    PPG                = mean(PointsFor, na.rm = TRUE),
    AvgPointDiff       = mean(PointDiff, na.rm = TRUE),
    SdPointDiff        = sd(PointDiff, na.rm = TRUE),
    
    # Momentum over the last 5 games inside the early window
    Last5WinPct = {
      w <- Win
      k <- min(5L, length(w))
      if (k > 0L) mean(tail(w, k)) else NA_real_
    },
    .groups = "drop"
  )

# ---- Merge schedule strength feature with original features ----
team_season_early <- team_season_early %>%
  left_join(schedule_strength, by = c("Season", "Team"))

# ---- Attach playoff label (madePlayoffs) ----
final_full <- team_season_early %>%
  left_join(playoff_teams, by = c("Season", "Team")) %>%
  mutate(
    madePlayoffs = if_else(is.na(madePlayoffs), 0L, 1L)
  ) %>%
  arrange(Season, Team)

# ---- Build TRIMMED table: ONLY the features + label + IDs (keeps Team here for splitting) ----
final_trimmed <- final_full %>%
  select(
    Season, Team,
    WinPct, HomeWinPct, BlowoutWinPct,
    AvgPointDiff, SdPointDiff, PPG, Last5WinPct,
    AvgOppWinPct,  # NEW SCHEDULE STRENGTH FEATURE
    madePlayoffs
  ) %>%
  arrange(Season, Team)

# ---- Helper: get season start year from 'Season' which may be "1999-00" or numeric ----
get_season_start <- function(x) {
  if (is.numeric(x)) return(as.integer(x))
  xs <- suppressWarnings(as.integer(substr(as.character(x), 1, 4)))
  na_idx <- is.na(xs)
  if (any(na_idx)) {
    xs[na_idx] <- suppressWarnings(as.integer(as.character(x[na_idx])))
  }
  xs
}

season_start <- get_season_start(final_trimmed$Season)

# ---- Split into TRAIN / TEST by 5-year era blocks with Train, Test, Train cycle ----
base_year <- 1947L
block_idx <- floor((season_start - base_year) / 5)
in_test_block  <- (block_idx %% 3L) == 1L
in_train_block <- !in_test_block

train_idx <- !is.na(season_start) & in_train_block
test_idx  <- !is.na(season_start) & in_test_block

# ---- Create TRAIN (nominal label, Season removed, Team removed) ----
train_df <- final_trimmed[train_idx, , drop = FALSE] %>%
  select(-Team) %>%
  mutate(madePlayoffs = as.integer(madePlayoffs)) %>%
  mutate(madePlayoffs = factor(ifelse(madePlayoffs == 1, "yes", "no"), levels = c("no","yes"))) %>%
  select(-Season)

# ---- Split TEST set (the rows that were assigned to TEST) into DEV and TEST by straight alternating rows ----
test_rows <- final_trimmed[test_idx, , drop = FALSE]

if (nrow(test_rows) == 0) {
  stop("No rows were assigned to the TEST era blocks (check my Season values bro or the 5-year blocking).")
}

# Keep the existing ordering (final_trimmed was arranged by Season, Team).
# Assign alternating rows: odd-index -> development, even-index -> test
idx <- seq_len(nrow(test_rows))
dev_in_test <- (idx %% 2L) == 1L   # 1,3,5,... -> development
test_in_test <- (idx %% 2L) == 0L  # 2,4,6,... -> test

dev_df <- test_rows[dev_in_test, , drop = FALSE] %>%
  select(-Team) %>%
  mutate(madePlayoffs = as.integer(madePlayoffs)) %>%
  mutate(madePlayoffs = factor(ifelse(madePlayoffs == 1, "yes", "no"), levels = c("no","yes"))) %>%
  select(-Season)

test_df <- test_rows[test_in_test, , drop = FALSE] %>%
  select(-Team) %>%
  mutate(madePlayoffs = as.integer(madePlayoffs)) %>%
  mutate(madePlayoffs = factor(ifelse(madePlayoffs == 1, "yes", "no"), levels = c("no","yes"))) %>%
  select(-Season)

# ---- Sanity checks: ensure each split has at least 150 instances ----
count_train <- nrow(train_df)
count_dev   <- nrow(dev_df)
count_test  <- nrow(test_df)

if (count_train < 150) {
  stop(sprintf("Train set has %d rows (<150). The era blocking produced too few train instances.", count_train))
}
if (count_dev < 150) {
  stop(sprintf("Development set has %d rows (<150). The alternating split of test produced too few dev instances.", count_dev))
}
if (count_test < 150) {
  stop(sprintf("Test set has %d rows (<150). The alternating split of test produced too few test instances.", count_test))
}

# ---- Write the three outputs (nominal labels, no Season) ----
write_csv(train_df, path_out_train_nom_noSeason)
write_csv(dev_df,   path_out_dev_nom_noSeason)
write_csv(test_df,  path_out_test_nom_noSeason)

# ---- Summary prints ----
cat(sprintf(" Wrote nominal TRAIN (no Season) -> %s with %d rows, %d cols.\n",
            path_out_train_nom_noSeason, nrow(train_df), ncol(train_df)))
cat(sprintf(" Wrote nominal DEVELOPMENT (no Season) -> %s with %d rows, %d cols.\n",
            path_out_dev_nom_noSeason, nrow(dev_df), ncol(dev_df)))
cat(sprintf(" Wrote nominal TEST (no Season) -> %s with %d rows, %d cols.\n",
            path_out_test_nom_noSeason, nrow(test_df), ncol(test_df)))

# ---- Feature Summary ----
cat("\n FEATURE SUMMARY:\n")
cat("Original features (7): WinPct, HomeWinPct, BlowoutWinPct, AvgPointDiff, SdPointDiff, PPG, Last5WinPct\n")
cat("NEW feature added (1): AvgOppWinPct (Average Opponent Win Percentage)\n")
cat(sprintf("Total features: %d\n", ncol(train_df) - 1))
cat(sprintf("Label: madePlayoffs (no/yes)\n\n"))

# ---- Sample Statistics for New Feature ----
cat("AvgOppWinPct STATISTICS (Schedule Strength):\n")
cat(sprintf("  Train - Mean: %.4f, SD: %.4f, Range: [%.4f, %.4f]\n",
            mean(train_df$AvgOppWinPct, na.rm = TRUE),
            sd(train_df$AvgOppWinPct, na.rm = TRUE),
            min(train_df$AvgOppWinPct, na.rm = TRUE),
            max(train_df$AvgOppWinPct, na.rm = TRUE)))

cat(sprintf("  Dev   - Mean: %.4f, SD: %.4f, Range: [%.4f, %.4f]\n",
            mean(dev_df$AvgOppWinPct, na.rm = TRUE),
            sd(dev_df$AvgOppWinPct, na.rm = TRUE),
            min(dev_df$AvgOppWinPct, na.rm = TRUE),
            max(dev_df$AvgOppWinPct, na.rm = TRUE)))

cat(sprintf("  Test  - Mean: %.4f, SD: %.4f, Range: [%.4f, %.4f]\n",
            mean(test_df$AvgOppWinPct, na.rm = TRUE),
            sd(test_df$AvgOppWinPct, na.rm = TRUE),
            min(test_df$AvgOppWinPct, na.rm = TRUE),
            max(test_df$AvgOppWinPct, na.rm = TRUE)))

cat("\n Interpretation:\n")
cat("   AvgOppWinPct ≈ 0.50: Faced average-strength opponents\n")
cat("   AvgOppWinPct > 0.55: Faced tough schedule (stats may be suppressed)\n")
cat("   AvgOppWinPct < 0.45: Faced weak schedule (stats may be inflated)\n")
