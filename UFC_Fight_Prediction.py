import pandas as pd
import numpy as np


# ============================================================
# PHASE 01 — PROJECT INITIALIZATION
# Subtitle: Environment Setup & Dataset Loading
# ============================================================

DATA_PATH = "data/raw/UFC_full_data_silver_v2.csv"

print("\n" + "=" * 70)
print("PHASE 01 — PROJECT INITIALIZATION")
print("Environment Setup & Dataset Loading")
print("=" * 70)

# Load dataset
data = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print(f"Rows    : {data.shape[0]:,}")
print(f"Columns : {data.shape[1]:,}")


# ============================================================
# PHASE 02 — DATA AUDIT
# Subtitle: Dataset Structure, Quality & Integrity
# ============================================================

print("\n" + "=" * 70)
print("PHASE 02 — DATA AUDIT")
print("Dataset Structure, Quality & Integrity")
print("=" * 70)


# ------------------------------------------------------------
# 2.1 Dataset structure
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("2.1 DATASET STRUCTURE")
print("-" * 60)

print(f"Rows    : {data.shape[0]:,}")
print(f"Columns : {data.shape[1]:,}")


# ------------------------------------------------------------
# 2.2 Columns
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("2.2 COLUMNS")
print("-" * 60)

for i, column in enumerate(data.columns, start=1):
    print(f"{i:>3}. {column}")


# ------------------------------------------------------------
# 2.3 Data types
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("2.3 DATA TYPES")
print("-" * 60)

print(data.dtypes)


# ------------------------------------------------------------
# 2.4 First rows
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("2.4 FIRST 5 ROWS")
print("-" * 60)

print(data.head().to_string())


# ------------------------------------------------------------
# 2.5 Missing values
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("2.5 MISSING VALUES")
print("-" * 60)

missing = data.isna().sum().to_frame("missing_count")

missing["missing_pct"] = (
    missing["missing_count"] / len(data) * 100
)

missing = missing.sort_values(
    "missing_pct",
    ascending=False
)

print(
    missing[
        missing["missing_count"] > 0
    ].to_string()
)


# ------------------------------------------------------------
# 2.6 Duplicate rows
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("2.6 DUPLICATE ROWS")
print("-" * 60)

duplicate_count = data.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")


# ------------------------------------------------------------
# 2.7 Target distribution
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("2.7 TARGET DISTRIBUTION")
print("-" * 60)

print(data["winner"].value_counts(dropna=False).to_string())

print("\nTarget percentages:")

print(
    (
        data["winner"]
        .value_counts(normalize=True, dropna=False)
        * 100
    )
    .round(2)
    .to_string()
)


# ------------------------------------------------------------
# 2.8 Fight result distribution
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("2.8 FIGHT RESULT DISTRIBUTION")
print("-" * 60)

print(
    data["result"]
    .value_counts(dropna=False)
    .to_string()
)


# ------------------------------------------------------------
# 2.9 Event date validation
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("2.9 EVENT DATE VALIDATION")
print("-" * 60)

event_dates = pd.to_datetime(
    data["event_date"],
    errors="coerce"
)

print("Earliest date     :", event_dates.min())
print("Latest date       :", event_dates.max())
print("Invalid dates     :", event_dates.isna().sum())
print("Unique event dates:", event_dates.nunique())


# ------------------------------------------------------------
# 2.10 Chronological order
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("2.10 CHRONOLOGICAL ORDER")
print("-" * 60)

print(
    "Already sorted chronologically:",
    event_dates.is_monotonic_increasing
)

print("\nFirst 5 dates:")
print(event_dates.head().to_string(index=False))

print("\nLast 5 dates:")
print(event_dates.tail().to_string(index=False))


# ============================================================
# PHASE 03 — EXPLORATORY DATA ANALYSIS
# Subtitle: Statistical Insights, Distributions & Data Patterns
# ============================================================

print("\n" + "=" * 70)
print("PHASE 03 — EXPLORATORY DATA ANALYSIS")
print("Statistical Insights, Distributions & Data Patterns")
print("=" * 70)


# ------------------------------------------------------------
# Prepare EDA dataset
# ------------------------------------------------------------

eda_data = data.copy()

eda_data["event_date"] = pd.to_datetime(
    eda_data["event_date"],
    errors="coerce"
)


# ------------------------------------------------------------
# 3.1 WEIGHT CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("3.1 WEIGHT CLASS DISTRIBUTION")
print("-" * 60)

weight_counts = (
    eda_data["weight_class"]
    .value_counts()
)

print(weight_counts.to_string())


# ------------------------------------------------------------
# 3.2 FIGHTS OVER TIME
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("3.2 FIGHTS OVER TIME")
print("-" * 60)

fights_per_year = (
    eda_data["event_date"]
    .dt.year
    .value_counts()
    .sort_index()
)

print(fights_per_year.to_string())


# ------------------------------------------------------------
# 3.3 FIGHT RESULT DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("3.3 FIGHT RESULT DISTRIBUTION")
print("-" * 60)

result_counts = (
    eda_data["result"]
    .value_counts()
)

print(result_counts.to_string())


# ------------------------------------------------------------
# 3.4 GENDER DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("3.4 GENDER DISTRIBUTION")
print("-" * 60)

gender_counts = (
    eda_data["gender"]
    .value_counts()
)

print(gender_counts.to_string())


# ------------------------------------------------------------
# 3.5 TITLE FIGHT DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("3.5 TITLE FIGHT DISTRIBUTION")
print("-" * 60)

title_counts = (
    eda_data["title_fight"]
    .value_counts()
)

print(title_counts.to_string())


# ------------------------------------------------------------
# 3.6 NUMBER OF ROUNDS
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("3.6 NUMBER OF ROUNDS")
print("-" * 60)

round_counts = (
    eda_data["num_rounds"]
    .value_counts(dropna=False)
    .sort_index()
)

print(round_counts.to_string())


# ------------------------------------------------------------
# 3.7 FIGHTER EXPERIENCE
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("3.7 FIGHTER EXPERIENCE")
print("-" * 60)

experience_data = pd.concat(
    [
        eda_data[
            [
                "f_1_name",
                "f_1_fighter_w",
                "f_1_fighter_l"
            ]
        ].rename(
            columns={
                "f_1_name": "fighter",
                "f_1_fighter_w": "wins",
                "f_1_fighter_l": "losses"
            }
        ),

        eda_data[
            [
                "f_2_name",
                "f_2_fighter_w",
                "f_2_fighter_l"
            ]
        ].rename(
            columns={
                "f_2_name": "fighter",
                "f_2_fighter_w": "wins",
                "f_2_fighter_l": "losses"
            }
        )
    ],
    ignore_index=True
)

experience_data["career_fights"] = (
    experience_data["wins"] +
    experience_data["losses"]
)

print(
    experience_data["career_fights"]
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# 3.8 HEIGHT vs WEIGHT
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("3.8 HEIGHT vs WEIGHT")
print("-" * 60)

physical_data = eda_data[
    [
        "f_1_fighter_height_cm",
        "f_1_fighter_weight_lbs"
    ]
].dropna()

print(
    physical_data.describe()
    .to_string()
)


# ------------------------------------------------------------
# 3.9 HEIGHT vs REACH
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("3.9 HEIGHT vs REACH")
print("-" * 60)

reach_data = eda_data[
    [
        "f_1_fighter_height_cm",
        "f_1_fighter_reach_cm"
    ]
].dropna()

print(
    reach_data.describe()
    .to_string()
)


# ------------------------------------------------------------
# 3.10 FIGHTER 1 vs FIGHTER 2 WIN DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("3.10 FIGHTER POSITION WIN DISTRIBUTION")
print("-" * 60)

winner_position = pd.Series(
    "Unknown",
    index=eda_data.index,
    dtype="object"
)

winner_position.loc[
    eda_data["winner"] == eda_data["f_1_name"]
] = "Fighter 1"

winner_position.loc[
    eda_data["winner"] == eda_data["f_2_name"]
] = "Fighter 2"

winner_position = winner_position.value_counts()

print(winner_position.to_string())

# ------------------------------------------------------------
# 3.11 TOP FIGHTERS BY APPEARANCES
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("3.11 TOP 20 FIGHTERS BY APPEARANCES")
print("-" * 60)

fighter_counts = pd.concat(
    [
        eda_data["f_1_name"],
        eda_data["f_2_name"]
    ]
).value_counts()

print(
    fighter_counts
    .head(20)
    .to_string()
)


# ------------------------------------------------------------
# 3.12 IMPORTANT FEATURES — MISSING VALUES
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("3.12 IMPORTANT FEATURES — MISSING VALUES")
print("-" * 60)

important_columns = [
    "winner",
    "result",
    "weight_class",
    "gender",
    "event_date",
    "f_1_name",
    "f_2_name",
    "f_1_fighter_height_cm",
    "f_2_fighter_height_cm",
    "f_1_fighter_weight_lbs",
    "f_2_fighter_weight_lbs",
    "f_1_fighter_reach_cm",
    "f_2_fighter_reach_cm",
    "f_1_fighter_stance",
    "f_2_fighter_stance",
    "f_1_fighter_w",
    "f_1_fighter_l",
    "f_2_fighter_w",
    "f_2_fighter_l",
]

missing_important = (
    eda_data[important_columns]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print(
    missing_important
    .to_string()
)


# ============================================================
# PHASE 03 SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 03 — EDA COMPLETE")
print("=" * 70)

print(f"\nTotal fights       : {len(eda_data):,}")

print(
    f"Unique fighters    : "
    f"{pd.concat([eda_data['f_1_name'], eda_data['f_2_name']]).nunique():,}"
)

print(
    f"Weight classes     : "
    f"{eda_data['weight_class'].nunique():,}"
)

print(
    f"Date range         : "
    f"{eda_data['event_date'].min().date()} "
    f"to "
    f"{eda_data['event_date'].max().date()}"
)

print("\nEDA pipeline completed successfully.")


# ============================================================
# PHASE 04 — DATA PREPROCESSING
# Subtitle: Cleaning, Target Validation & Model-Ready Dataset
# ============================================================

print("\n" + "=" * 70)
print("PHASE 04 — DATA PREPROCESSING")
print("Cleaning, Target Validation & Model-Ready Dataset")
print("=" * 70)


# ------------------------------------------------------------
# 4.1 CREATE WORKING DATASET
# ------------------------------------------------------------

model_data = data.copy()

print("\n" + "-" * 60)
print("4.1 WORKING DATASET")
print("-" * 60)

print(f"Original rows    : {len(model_data):,}")
print(f"Original columns : {model_data.shape[1]:,}")


# ------------------------------------------------------------
# 4.2 CONVERT EVENT DATE
# ------------------------------------------------------------

model_data["event_date"] = pd.to_datetime(
    model_data["event_date"],
    errors="coerce"
)

print("\n" + "-" * 60)
print("4.2 EVENT DATE")
print("-" * 60)

print(
    "Invalid dates:",
    model_data["event_date"].isna().sum()
)

print(
    "Date range:",
    model_data["event_date"].min(),
    "to",
    model_data["event_date"].max()
)


# ------------------------------------------------------------
# 4.3 SORT CHRONOLOGICALLY
# ------------------------------------------------------------

model_data = (
    model_data
    .sort_values("event_date")
    .reset_index(drop=True)
)

print("\n" + "-" * 60)
print("4.3 CHRONOLOGICAL SORT")
print("-" * 60)

print(
    "Sorted chronologically:",
    model_data["event_date"].is_monotonic_increasing
)


# ------------------------------------------------------------
# 4.4 CHECK MISSING TARGETS
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("4.4 TARGET AVAILABILITY")
print("-" * 60)

missing_winner = model_data["winner"].isna().sum()

print("Missing winner:", missing_winner)

print(
    "Missing winner percentage:",
    round(missing_winner / len(model_data) * 100, 2),
    "%"
)


# ------------------------------------------------------------
# 4.5 REMOVE ROWS WITHOUT TARGET
# ------------------------------------------------------------

before = len(model_data)

model_data = model_data[
    model_data["winner"].notna()
].copy()

after = len(model_data)

print("\n" + "-" * 60)
print("4.5 REMOVE UNKNOWN TARGETS")
print("-" * 60)

print("Rows before :", before)
print("Rows removed:", before - after)
print("Rows after  :", after)


# ------------------------------------------------------------
# 4.6 FIGHTER NAME VALIDATION
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("4.6 FIGHTER NAME VALIDATION")
print("-" * 60)

missing_f1 = model_data["f_1_name"].isna().sum()
missing_f2 = model_data["f_2_name"].isna().sum()

print("Missing Fighter 1:", missing_f1)
print("Missing Fighter 2:", missing_f2)

unique_fighters = pd.concat(
    [
        model_data["f_1_name"],
        model_data["f_2_name"]
    ]
).nunique()

print("Unique fighters:", unique_fighters)


# ------------------------------------------------------------
# 4.7 VALIDATE WINNER
# ------------------------------------------------------------

valid_winner = (
    (model_data["winner"] == model_data["f_1_name"]) |
    (model_data["winner"] == model_data["f_2_name"])
)

invalid_winner = ~valid_winner

print("\n" + "-" * 60)
print("4.7 WINNER CONSISTENCY")
print("-" * 60)

print(
    "Valid winner:",
    valid_winner.sum()
)

print(
    "Invalid winner:",
    invalid_winner.sum()
)

if invalid_winner.sum() > 0:

    print("\nExamples of invalid winner rows:")

    print(
        model_data.loc[
            invalid_winner,
            [
                "f_1_name",
                "f_2_name",
                "winner",
                "result"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# 4.8 KEEP ONLY VALID FIGHTS
# ------------------------------------------------------------

model_data = model_data[
    valid_winner
].copy()

print("\n" + "-" * 60)
print("4.8 VALID FIGHTS")
print("-" * 60)

print(
    "Rows after winner validation:",
    len(model_data)
)


# ------------------------------------------------------------
# 4.9 CREATE BINARY TARGET
# ------------------------------------------------------------

model_data["target"] = (
    model_data["winner"] == model_data["f_1_name"]
).astype(int)

print("\n" + "-" * 60)
print("4.9 BINARY TARGET")
print("-" * 60)

print("Target definition:")
print("1 = Fighter 1 wins")
print("0 = Fighter 2 wins")

print("\nTarget distribution:")

print(
    model_data["target"]
    .value_counts()
    .sort_index()
)

print("\nTarget percentages:")

print(
    (
        model_data["target"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )
    .round(2)
)


# ------------------------------------------------------------
# 4.10 CHECK FIGHTER DUPLICATES
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("4.10 SAME FIGHTER CHECK")
print("-" * 60)

same_fighter = (
    model_data["f_1_name"] ==
    model_data["f_2_name"]
)

print(
    "Rows with identical Fighter 1 and Fighter 2:",
    same_fighter.sum()
)

if same_fighter.sum() > 0:

    print("\nExamples:")

    print(
        model_data.loc[
            same_fighter,
            [
                "event_name",
                "f_1_name",
                "f_2_name",
                "winner"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# 4.11 CHECK DUPLICATES AFTER CLEANING
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("4.11 DUPLICATE CHECK")
print("-" * 60)

print(
    "Duplicate rows:",
    model_data.duplicated().sum()
)


# ------------------------------------------------------------
# 4.12 IMPORTANT NUMERICAL FEATURES
# ------------------------------------------------------------

numerical_features = [
    "f_1_fighter_height_cm",
    "f_2_fighter_height_cm",
    "f_1_fighter_weight_lbs",
    "f_2_fighter_weight_lbs",
    "f_1_fighter_reach_cm",
    "f_2_fighter_reach_cm",
    "f_1_fighter_w",
    "f_1_fighter_l",
    "f_1_fighter_d",
    "f_2_fighter_w",
    "f_2_fighter_l",
    "f_2_fighter_d",
]

print("\n" + "-" * 60)
print("4.12 NUMERICAL FEATURES")
print("-" * 60)

available_numerical = [
    column
    for column in numerical_features
    if column in model_data.columns
]

print(
    "Available numerical features:",
    len(available_numerical)
)

print(available_numerical)


# ------------------------------------------------------------
# 4.13 MISSING VALUES AFTER CLEANING
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("4.13 MISSING VALUES")
print("-" * 60)

missing_after = (
    model_data
    .isna()
    .sum()
    .sort_values(ascending=False)
)

missing_after = missing_after[
    missing_after > 0
]

print(
    missing_after
    .head(30)
    .to_string()
)


# ------------------------------------------------------------
# 4.14 FINAL DATASET SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 04 SUMMARY")
print("=" * 70)

print(
    f"Final rows       : {model_data.shape[0]:,}"
)

print(
    f"Final columns    : {model_data.shape[1]:,}"
)

print(
    f"Unique fighters  : "
    f"{pd.concat([model_data['f_1_name'], model_data['f_2_name']]).nunique():,}"
)

print(
    "Date range:",
    model_data["event_date"].min(),
    "to",
    model_data["event_date"].max()
)

print(
    f"Missing targets  : {model_data['target'].isna().sum()}"
)

print(
    f"Duplicate rows   : {model_data.duplicated().sum()}"
)

print("\nPhase 04 completed successfully.")

# ============================================================
# PHASE 05 — FEATURE ENGINEERING
# Subtitle: Leakage-Safe Pre-Fight Fighter Features
# ============================================================

print("\n" + "=" * 70)
print("PHASE 05 — FEATURE ENGINEERING")
print("Leakage-Safe Pre-Fight Fighter Features")
print("=" * 70)


# ------------------------------------------------------------
# 5.1 CREATE FEATURE ENGINEERING DATASET
# ------------------------------------------------------------

feature_data = model_data.copy()

# Make sure fights are strictly chronological
feature_data = (
    feature_data
    .sort_values("event_date")
    .reset_index(drop=True)
)

print("\n" + "-" * 60)
print("5.1 FEATURE ENGINEERING DATASET")
print("-" * 60)

print(f"Rows    : {feature_data.shape[0]:,}")
print(f"Columns : {feature_data.shape[1]:,}")


# ------------------------------------------------------------
# 5.2 INITIALIZE HISTORICAL FIGHTER RECORD
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("5.2 HISTORICAL FIGHTER RECORD")
print("-" * 60)

fighter_history = {}

print("Historical fighter tracking initialized.")


# ------------------------------------------------------------
# 5.3 FUNCTION TO GET PRE-FIGHT HISTORY
# ------------------------------------------------------------

def get_fighter_history(fighter_name):
    """
    Return fighter statistics accumulated before the current fight.
    """

    if fighter_name not in fighter_history:
        return {
            "wins": 0,
            "losses": 0,
            "draws": 0
        }

    return fighter_history[fighter_name].copy()


# ------------------------------------------------------------
# 5.4 CREATE PRE-FIGHT FEATURES
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("5.4 BUILD PRE-FIGHT FEATURES")
print("-" * 60)

pre_fight_features = []


for _, row in feature_data.iterrows():

    fighter_1 = row["f_1_name"]
    fighter_2 = row["f_2_name"]

    # --------------------------------------------------------
    # Get history BEFORE current fight
    # --------------------------------------------------------

    f1_history = get_fighter_history(fighter_1)
    f2_history = get_fighter_history(fighter_2)

    # --------------------------------------------------------
    # Calculate total fights BEFORE current fight
    # --------------------------------------------------------

    f1_total_fights = (
        f1_history["wins"] +
        f1_history["losses"] +
        f1_history["draws"]
    )

    f2_total_fights = (
        f2_history["wins"] +
        f2_history["losses"] +
        f2_history["draws"]
    )

    # --------------------------------------------------------
    # Calculate win rate BEFORE current fight
    # --------------------------------------------------------

    f1_win_rate = (
        f1_history["wins"] / f1_total_fights
        if f1_total_fights > 0
        else 0.0
    )

    f2_win_rate = (
        f2_history["wins"] / f2_total_fights
        if f2_total_fights > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Store pre-fight features
    # --------------------------------------------------------

    pre_fight_features.append({

        # Fighter identity
        "f_1_name": fighter_1,
        "f_2_name": fighter_2,

        # Fighter 1 historical record
        "f_1_previous_wins": f1_history["wins"],
        "f_1_previous_losses": f1_history["losses"],
        "f_1_previous_draws": f1_history["draws"],
        "f_1_previous_fights": f1_total_fights,
        "f_1_win_rate": f1_win_rate,

        # Fighter 2 historical record
        "f_2_previous_wins": f2_history["wins"],
        "f_2_previous_losses": f2_history["losses"],
        "f_2_previous_draws": f2_history["draws"],
        "f_2_previous_fights": f2_total_fights,
        "f_2_win_rate": f2_win_rate,

        # Historical differences
        "win_rate_diff": f1_win_rate - f2_win_rate,
        "experience_diff": f1_total_fights - f2_total_fights,
        "wins_diff": (
            f1_history["wins"] -
            f2_history["wins"]
        ),
        "losses_diff": (
            f1_history["losses"] -
            f2_history["losses"]
        ),
        "draws_diff": (
            f1_history["draws"] -
            f2_history["draws"]
        ),
    })

    # --------------------------------------------------------
    # UPDATE HISTORY AFTER current fight
    # --------------------------------------------------------

    winner = row["winner"]

    if fighter_1 not in fighter_history:
        fighter_history[fighter_1] = {
            "wins": 0,
            "losses": 0,
            "draws": 0
        }

    if fighter_2 not in fighter_history:
        fighter_history[fighter_2] = {
            "wins": 0,
            "losses": 0,
            "draws": 0
        }

    # Fighter 1 wins
    if winner == fighter_1:

        fighter_history[fighter_1]["wins"] += 1
        fighter_history[fighter_2]["losses"] += 1

    # Fighter 2 wins
    elif winner == fighter_2:

        fighter_history[fighter_2]["wins"] += 1
        fighter_history[fighter_1]["losses"] += 1


# ------------------------------------------------------------
# 5.5 CONVERT FEATURES TO DATAFRAME
# ------------------------------------------------------------

historical_features = pd.DataFrame(
    pre_fight_features
)

print(
    "Historical feature rows:",
    len(historical_features)
)

print(
    "Historical feature columns:",
    len(historical_features.columns)
)


# ------------------------------------------------------------
# 5.6 MERGE FEATURES WITH FIGHT DATA
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("5.6 MERGE HISTORICAL FEATURES")
print("-" * 60)

# Drop identity columns from feature table before merge
historical_features_for_merge = historical_features.drop(
    columns=["f_1_name", "f_2_name"]
)

feature_data = pd.concat(
    [
        feature_data.reset_index(drop=True),
        historical_features_for_merge.reset_index(drop=True)
    ],
    axis=1
)

print(
    f"Dataset shape after feature engineering: "
    f"{feature_data.shape}"
)


# ------------------------------------------------------------
# 5.7 CHECK FIRST FIGHTS
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("5.7 FIRST FIGHTS CHECK")
print("-" * 60)

check_columns = [
    "event_date",
    "f_1_name",
    "f_2_name",
    "target",
    "f_1_previous_fights",
    "f_2_previous_fights",
    "f_1_win_rate",
    "f_2_win_rate",
    "win_rate_diff",
    "experience_diff"
]

print(
    feature_data[check_columns]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 5.8 CHECK HISTORICAL FEATURE RANGES
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("5.8 HISTORICAL FEATURE SUMMARY")
print("-" * 60)

historical_columns = [
    "f_1_previous_wins",
    "f_1_previous_losses",
    "f_1_previous_draws",
    "f_1_previous_fights",
    "f_1_win_rate",
    "f_2_previous_wins",
    "f_2_previous_losses",
    "f_2_previous_draws",
    "f_2_previous_fights",
    "f_2_win_rate",
    "win_rate_diff",
    "experience_diff",
    "wins_diff",
    "losses_diff",
    "draws_diff"
]

print(
    feature_data[historical_columns]
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# 5.9 LEAKAGE CHECK
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("5.9 LEAKAGE CHECK")
print("-" * 60)

# First fight for a fighter should have zero previous fights.
first_fight_mask = (
    feature_data["f_1_previous_fights"] == 0
)

print(
    "Rows where Fighter 1 had no previous fights:",
    first_fight_mask.sum()
)

print(
    "Rows where Fighter 2 had no previous fights:",
    (
        feature_data["f_2_previous_fights"] == 0
    ).sum()
)

# Check that historical win rates stay between 0 and 1
invalid_f1_rates = (
    (feature_data["f_1_win_rate"] < 0) |
    (feature_data["f_1_win_rate"] > 1)
).sum()

invalid_f2_rates = (
    (feature_data["f_2_win_rate"] < 0) |
    (feature_data["f_2_win_rate"] > 1)
).sum()

print(
    "Invalid Fighter 1 win rates:",
    invalid_f1_rates
)

print(
    "Invalid Fighter 2 win rates:",
    invalid_f2_rates
)


# ------------------------------------------------------------
# 5.10 FINAL FEATURE DATASET
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 05 SUMMARY")
print("=" * 70)

print(
    f"Rows              : {feature_data.shape[0]:,}"
)

print(
    f"Columns            : {feature_data.shape[1]:,}"
)

print(
    f"Historical features: {len(historical_columns):,}"
)

print(
    "Temporal ordering  :",
    feature_data["event_date"].is_monotonic_increasing
)

print(
    "Target missing     :",
    feature_data["target"].isna().sum()
)

print("\nPhase 05 completed successfully.")

# ============================================================
# PHASE 06 — DATASET PREPARATION
# Target Definition, Fighter Perspective & Temporal Split
# ============================================================

print("\n" + "=" * 70)
print("PHASE 06 — DATASET PREPARATION")
print("Target Definition, Fighter Perspective & Temporal Split")
print("=" * 70)


# ------------------------------------------------------------
# 6.1 COPY FEATURE DATA
# ------------------------------------------------------------

modeling_data = feature_data.copy()

print("\n6.1 MODELING DATA")
print("-" * 60)

print("Rows:", len(modeling_data))
print("Columns:", len(modeling_data.columns))


# ------------------------------------------------------------
# 6.2 BASIC VALIDATION
# ------------------------------------------------------------

required_columns = [
    "event_date",
    "winner",
    "f_1_name",
    "f_2_name"
]

missing_required = [
    column
    for column in required_columns
    if column not in modeling_data.columns
]

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )


# ------------------------------------------------------------
# 6.3 SORT CHRONOLOGICALLY
# ------------------------------------------------------------

print("\n6.3 CHRONOLOGICAL ORDER")
print("-" * 60)

modeling_data["event_date"] = pd.to_datetime(
    modeling_data["event_date"],
    errors="coerce"
)

modeling_data = (
    modeling_data
    .sort_values("event_date")
    .reset_index(drop=True)
)

print(
    "Date range:",
    modeling_data["event_date"].min(),
    "to",
    modeling_data["event_date"].max()
)


# ------------------------------------------------------------
# 6.4 REMOVE INVALID WINNER ROWS
# ------------------------------------------------------------

print("\n6.4 WINNER VALIDATION")
print("-" * 60)

valid_winner = (
    modeling_data["winner"] == modeling_data["f_1_name"]
)

invalid_rows = (~valid_winner).sum()

print("Invalid winner rows:", invalid_rows)

if invalid_rows > 0:
    print(
        "WARNING: Some rows do not follow the expected "
        "winner-first structure."
    )

    modeling_data = modeling_data.loc[
        valid_winner
    ].copy()

    modeling_data = (
        modeling_data
        .sort_values("event_date")
        .reset_index(drop=True)
    )

print(
    "Rows after validation:",
    len(modeling_data)
)


# ------------------------------------------------------------
# 6.5 CREATE TARGET
# ------------------------------------------------------------

print("\n6.5 ORIGINAL TARGET")
print("-" * 60)

# In the raw dataset:
# f_1 = winner
# f_2 = loser
#
# Therefore, before randomizing perspective:
#
# f_1 -> target 1
# f_2 -> target 0

modeling_data["original_target"] = 1

print(
    "Original target classes:",
    modeling_data["original_target"].unique().tolist()
)


# ------------------------------------------------------------
# 6.6 RANDOMIZE FIGHTER PERSPECTIVE
# ------------------------------------------------------------

print("\n6.6 FIGHTER PERSPECTIVE RANDOMIZATION")
print("-" * 60)

# IMPORTANT:
#
# The raw dataset always stores:
#
#     f_1 = winner
#     f_2 = loser
#
# This creates structural target leakage.
#
# We therefore randomly flip the perspective of each fight.
#
# If flip = False:
#
#     Fighter 1 = original winner
#     Fighter 2 = original loser
#     target = 1
#
# If flip = True:
#
#     Fighter 1 = original loser
#     Fighter 2 = original winner
#     target = 0
#
# All f_1_* and f_2_* columns are swapped together.

rng = np.random.RandomState(42)

modeling_data["flip_fighters"] = (
    rng.rand(len(modeling_data)) < 0.50
)

print(
    "Original perspective:",
    (~modeling_data["flip_fighters"]).sum()
)

print(
    "Flipped perspective:",
    modeling_data["flip_fighters"].sum()
)


# ------------------------------------------------------------
# 6.7 IDENTIFY FIGHTER-SPECIFIC COLUMNS
# ------------------------------------------------------------

print("\n6.7 FIGHTER COLUMN IDENTIFICATION")
print("-" * 60)

f1_columns = [
    column
    for column in modeling_data.columns
    if column.startswith("f_1_")
]

f2_columns = [
    column
    for column in modeling_data.columns
    if column.startswith("f_2_")
]

print("Fighter 1 columns:", len(f1_columns))
print("Fighter 2 columns:", len(f2_columns))


# ------------------------------------------------------------
# 6.8 BUILD F1/F2 COLUMN PAIRS
# ------------------------------------------------------------

column_pairs = []

for f1_column in f1_columns:

    corresponding_f2_column = (
        "f_2_" + f1_column[len("f_1_"):]
    )

    if corresponding_f2_column in modeling_data.columns:
        column_pairs.append(
            (f1_column, corresponding_f2_column)
        )

print(
    "Matched F1/F2 pairs:",
    len(column_pairs)
)


# ------------------------------------------------------------
# 6.9 SWAP FIGHTER INFORMATION
# ------------------------------------------------------------

print("\n6.9 SWAPPING FIGHTER INFORMATION")
print("-" * 60)

for f1_column, f2_column in column_pairs:

    temp = modeling_data[f1_column].copy()

    modeling_data.loc[
        modeling_data["flip_fighters"],
        f1_column
    ] = modeling_data.loc[
        modeling_data["flip_fighters"],
        f2_column
    ]

    modeling_data.loc[
        modeling_data["flip_fighters"],
        f2_column
    ] = modeling_data.loc[
        modeling_data["flip_fighters"],
        f1_column
    ]

    # IMPORTANT:
    # The previous assignment above can overwrite values.
    # Therefore restore using the original temporary values
    # for flipped rows.

    modeling_data.loc[
        modeling_data["flip_fighters"],
        f2_column
    ] = temp.loc[
        modeling_data["flip_fighters"]
    ]


# ------------------------------------------------------------
# 6.10 CREATE FINAL TARGET
# ------------------------------------------------------------

print("\n6.10 FINAL TARGET")
print("-" * 60)

# If the perspective was flipped:
#
# original winner moved to Fighter 2
# therefore Fighter 1 loses -> target 0
#
# Otherwise:
#
# original winner remains Fighter 1
# therefore target 1

modeling_data["target"] = np.where(
    modeling_data["flip_fighters"],
    0,
    1
)

modeling_data["target"] = (
    modeling_data["target"]
    .astype(int)
)

print(
    modeling_data["target"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nTarget proportions:")

print(
    modeling_data["target"]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
    .to_string()
)


# ------------------------------------------------------------
# 6.11 TARGET LEAKAGE CHECK
# ------------------------------------------------------------

print("\n6.11 TARGET LEAKAGE CHECK")
print("-" * 60)

# After the transformation:
#
# target = 1 -> current f_1 is original winner
# target = 0 -> current f_1 is original loser
#
# We verify that f_1 is NOT always the winner anymore.

current_f1_is_original_winner = (
    modeling_data["target"] == 1
)

current_f1_is_original_loser = (
    modeling_data["target"] == 0
)

print(
    "Current Fighter 1 = original winner:",
    current_f1_is_original_winner.sum()
)

print(
    "Current Fighter 1 = original loser:",
    current_f1_is_original_loser.sum()
)

if (
    current_f1_is_original_winner.sum()
    == len(modeling_data)
):
    raise ValueError(
        "TARGET LEAKAGE: Fighter 1 is still always the winner."
    )


# ------------------------------------------------------------
# 6.12 FEATURE SELECTION
# ------------------------------------------------------------

print("\n6.12 FEATURE SELECTION")
print("-" * 60)

historical_features = [
    "f_1_previous_wins",
    "f_1_previous_losses",
    "f_1_previous_draws",
    "f_1_previous_fights",
    "f_1_previous_win_rate",

    "f_2_previous_wins",
    "f_2_previous_losses",
    "f_2_previous_draws",
    "f_2_previous_fights",
    "f_2_previous_win_rate",

    "win_rate_diff",
    "experience_diff",
    "wins_diff",
    "losses_diff",
    "draws_diff"
]


physical_features = [
    "f_1_fighter_height_cm",
    "f_2_fighter_height_cm",

    "f_1_fighter_weight_lbs",
    "f_2_fighter_weight_lbs",

    "f_1_fighter_reach_cm",
    "f_2_fighter_reach_cm"
]


context_features = [
    "weight_class",
    "gender",
    "num_rounds",
    "title_fight"
]


stance_features = [
    "f_1_fighter_stance",
    "f_2_fighter_stance"
]


candidate_features = (
    historical_features
    + physical_features
    + context_features
    + stance_features
)


available_features = [
    column
    for column in candidate_features
    if column in modeling_data.columns
]


missing_features = [
    column
    for column in candidate_features
    if column not in modeling_data.columns
]


print(
    "Available features:",
    len(available_features)
)


if missing_features:

    print("\nMissing features:")

    for column in missing_features:
        print(" -", column)


# ------------------------------------------------------------
# 6.13 CREATE X AND y
# ------------------------------------------------------------

print("\n6.13 CREATE X AND y")
print("-" * 60)

X = modeling_data[
    available_features
].copy()

y = modeling_data[
    "target"
].copy()

print("X shape:", X.shape)
print("y shape:", y.shape)


# ------------------------------------------------------------
# 6.14 TEMPORAL SPLIT
# ------------------------------------------------------------

print("\n6.14 TEMPORAL SPLIT")
print("-" * 60)

n_rows = len(modeling_data)

train_end = int(n_rows * 0.70)

validation_end = int(n_rows * 0.85)


train_data = modeling_data.iloc[
    :train_end
].copy()


validation_data = modeling_data.iloc[
    train_end:validation_end
].copy()


test_data = modeling_data.iloc[
    validation_end:
].copy()


# ------------------------------------------------------------
# 6.15 TRAIN / VALIDATION / TEST
# ------------------------------------------------------------

X_train = train_data[
    available_features
].copy()

y_train = train_data[
    "target"
].copy()


X_validation = validation_data[
    available_features
].copy()

y_validation = validation_data[
    "target"
].copy()


X_test = test_data[
    available_features
].copy()

y_test = test_data[
    "target"
].copy()


# ------------------------------------------------------------
# 6.16 SPLIT SUMMARY
# ------------------------------------------------------------

print("\n6.16 SPLIT SUMMARY")
print("-" * 60)

print("\nTRAIN")

print("Rows:", len(train_data))

print(
    "Date:",
    train_data["event_date"].min(),
    "to",
    train_data["event_date"].max()
)

print("Target distribution:")

print(
    y_train
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nVALIDATION")

print("Rows:", len(validation_data))

print(
    "Date:",
    validation_data["event_date"].min(),
    "to",
    validation_data["event_date"].max()
)

print("Target distribution:")

print(
    y_validation
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nTEST")

print("Rows:", len(test_data))

print(
    "Date:",
    test_data["event_date"].min(),
    "to",
    test_data["event_date"].max()
)

print("Target distribution:")

print(
    y_test
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# 6.17 CLASS VALIDATION
# ------------------------------------------------------------

print("\n6.17 CLASS VALIDATION")
print("-" * 60)

train_classes = sorted(
    y_train.unique().tolist()
)

validation_classes = sorted(
    y_validation.unique().tolist()
)

test_classes = sorted(
    y_test.unique().tolist()
)

print("Train classes:", train_classes)

print(
    "Validation classes:",
    validation_classes
)

print("Test classes:", test_classes)


if len(train_classes) < 2:

    raise ValueError(
        "Training data contains only one class."
    )


# ------------------------------------------------------------
# 6.18 MISSING VALUES
# ------------------------------------------------------------

print("\n6.18 MISSING VALUES")
print("-" * 60)

missing_summary = (
    X_train
    .isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)

missing_summary = missing_summary[
    missing_summary > 0
]

if len(missing_summary) > 0:

    print(
        missing_summary.to_string()
    )

else:

    print(
        "No missing values in training features."
    )


# ------------------------------------------------------------
# 6.19 FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 06 COMPLETED")
print("=" * 70)

print(
    "Modeling rows:",
    len(modeling_data)
)

print(
    "Features:",
    len(available_features)
)

print(
    "Train rows:",
    len(X_train)
)

print(
    "Validation rows:",
    len(X_validation)
)

print(
    "Test rows:",
    len(X_test)
)

print(
    "\nFinal target distribution:"
)

print(
    modeling_data[
        "target"
    ]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
    .to_string()
)

print("\nReady for Phase 07.")

# ============================================================
# PHASE 07 — MODEL DEVELOPMENT
# Baseline Models & Algorithm Comparison
# ============================================================

print("\n" + "=" * 70)
print("PHASE 07 — MODEL DEVELOPMENT")
print("Baseline Models & Algorithm Comparison")
print("=" * 70)


# ------------------------------------------------------------
# 7.1 IMPORTS
# ------------------------------------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ------------------------------------------------------------
# 7.2 FEATURE NAME VALIDATION
# ------------------------------------------------------------

print("\n7.2 FEATURE VALIDATION")
print("-" * 60)

# Phase 05 uses these names.
# We explicitly use the names that actually exist.

historical_features = [
    "f_1_previous_wins",
    "f_1_previous_losses",
    "f_1_previous_draws",
    "f_1_previous_fights",
    "f_1_win_rate",

    "f_2_previous_wins",
    "f_2_previous_losses",
    "f_2_previous_draws",
    "f_2_previous_fights",
    "f_2_win_rate",

    "win_rate_diff",
    "experience_diff",
    "wins_diff",
    "losses_diff",
    "draws_diff"
]


physical_features = [
    "f_1_fighter_height_cm",
    "f_2_fighter_height_cm",
    "f_1_fighter_weight_lbs",
    "f_2_fighter_weight_lbs",
    "f_1_fighter_reach_cm",
    "f_2_fighter_reach_cm"
]


context_features = [
    "weight_class",
    "gender",
    "num_rounds",
    "title_fight"
]


stance_features = [
    "f_1_fighter_stance",
    "f_2_fighter_stance"
]


candidate_features = (
    historical_features
    + physical_features
    + context_features
    + stance_features
)


available_model_features = [
    column
    for column in candidate_features
    if column in X_train.columns
]


missing_model_features = [
    column
    for column in candidate_features
    if column not in X_train.columns
]


print(
    "Available model features:",
    len(available_model_features)
)


if missing_model_features:

    print("\nMissing model features:")

    for column in missing_model_features:
        print(" -", column)


# ------------------------------------------------------------
# 7.3 UPDATE DATASETS
# ------------------------------------------------------------

X_train_model = X_train[
    available_model_features
].copy()

X_validation_model = X_validation[
    available_model_features
].copy()

X_test_model = X_test[
    available_model_features
].copy()


# ------------------------------------------------------------
# 7.4 CATEGORICAL / NUMERICAL FEATURES
# ------------------------------------------------------------

print("\n7.4 FEATURE TYPES")
print("-" * 60)

categorical_features = [
    "weight_class",
    "gender",
    "f_1_fighter_stance",
    "f_2_fighter_stance"
]

categorical_features = [
    column
    for column in categorical_features
    if column in available_model_features
]


numerical_features = [
    column
    for column in available_model_features
    if column not in categorical_features
]


print(
    "Categorical features:",
    len(categorical_features)
)

for column in categorical_features:
    print(" -", column)


print(
    "\nNumerical features:",
    len(numerical_features)
)

for column in numerical_features:
    print(" -", column)


# ------------------------------------------------------------
# 7.5 VALIDATE NUMERICAL DATA
# ------------------------------------------------------------

print("\n7.5 NUMERICAL DATA VALIDATION")
print("-" * 60)

for column in numerical_features:

    if not pd.api.types.is_numeric_dtype(
        X_train_model[column]
    ):

        raise TypeError(
            f"Feature '{column}' is not numeric."
        )

print(
    "All numerical features are valid."
)


# ------------------------------------------------------------
# 7.6 PREPROCESSING PIPELINE
# ------------------------------------------------------------

print("\n7.6 PREPROCESSING PIPELINE")
print("-" * 60)

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ------------------------------------------------------------
# 7.7 DEFINE MODELS
# ------------------------------------------------------------

print("\n7.7 MODEL DEFINITIONS")
print("-" * 60)

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ------------------------------------------------------------
# 7.8 TRAIN MODELS
# ------------------------------------------------------------

print("\n7.8 MODEL TRAINING")
print("-" * 60)

trained_models = {}

validation_results = []


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train_model,
        y_train
    )

    trained_models[
        model_name
    ] = pipeline

    print(
        f"{model_name} training completed."
    )


# ------------------------------------------------------------
# 7.9 VALIDATION EVALUATION
# ------------------------------------------------------------

print("\n7.9 VALIDATION EVALUATION")
print("-" * 60)


for model_name, pipeline in trained_models.items():

    y_pred = pipeline.predict(
        X_validation_model
    )

    y_proba = pipeline.predict_proba(
        X_validation_model
    )[:, 1]


    accuracy = accuracy_score(
        y_validation,
        y_pred
    )

    precision = precision_score(
        y_validation,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_validation,
        y_proba
    )


    validation_results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }
    )


# ------------------------------------------------------------
# 7.10 RESULTS TABLE
# ------------------------------------------------------------

print("\n7.10 MODEL COMPARISON")
print("-" * 60)

results_df = pd.DataFrame(
    validation_results
)

results_df = results_df.sort_values(
    "roc_auc",
    ascending=False
).reset_index(drop=True)


print(
    results_df.round(4).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 7.11 MAJORITY CLASS BASELINE
# ------------------------------------------------------------

print("\n7.11 MAJORITY CLASS BASELINE")
print("-" * 60)

majority_class = (
    y_train
    .value_counts()
    .idxmax()
)

baseline_predictions = np.full(
    len(y_validation),
    majority_class
)

baseline_accuracy = accuracy_score(
    y_validation,
    baseline_predictions
)

print(
    "Majority class:",
    majority_class
)

print(
    "Baseline accuracy:",
    round(
        baseline_accuracy,
        4
    )
)


# ------------------------------------------------------------
# 7.12 SELECT BEST MODEL
# ------------------------------------------------------------

print("\n7.12 BEST MODEL")
print("-" * 60)

best_model_name = (
    results_df
    .iloc[0]["model"]
)

best_validation_auc = (
    results_df
    .iloc[0]["roc_auc"]
)

best_model = trained_models[
    best_model_name
]


print(
    "Best model:",
    best_model_name
)

print(
    "Validation ROC-AUC:",
    round(
        best_validation_auc,
        4
    )
)


# ------------------------------------------------------------
# 7.13 VALIDATION PREDICTIONS
# ------------------------------------------------------------

print("\n7.13 BEST MODEL VALIDATION")
print("-" * 60)

best_validation_predictions = (
    best_model.predict(
        X_validation_model
    )
)

best_validation_probabilities = (
    best_model.predict_proba(
        X_validation_model
    )[:, 1]
)


print(
    "Validation Accuracy:",
    round(
        accuracy_score(
            y_validation,
            best_validation_predictions
        ),
        4
    )
)

print(
    "Validation Precision:",
    round(
        precision_score(
            y_validation,
            best_validation_predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "Validation Recall:",
    round(
        recall_score(
            y_validation,
            best_validation_predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "Validation F1:",
    round(
        f1_score(
            y_validation,
            best_validation_predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "Validation ROC-AUC:",
    round(
        roc_auc_score(
            y_validation,
            best_validation_probabilities
        ),
        4
    )
)


# ------------------------------------------------------------
# 7.14 TEST SET PROTECTION
# ------------------------------------------------------------

print("\n7.14 TEST SET PROTECTION")
print("-" * 60)

print(
    "Test set was NOT used for model selection."
)

print(
    "Test set will remain untouched until Phase 08."
)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 07 COMPLETED")
print("=" * 70)

print(
    "Models trained:",
    len(trained_models)
)

print(
    "Best model:",
    best_model_name
)

print(
    "Best validation ROC-AUC:",
    round(
        best_validation_auc,
        4
    )
)

# ============================================================
# PHASE 07.5 — DATA LEAKAGE AUDIT
# Pre-Fight Feature Integrity Check
# ============================================================

print("\n" + "=" * 70)
print("PHASE 07.5 — DATA LEAKAGE AUDIT")
print("Pre-Fight Feature Integrity Check")
print("=" * 70)


# ------------------------------------------------------------
# 7.5.1 FEATURE LIST
# ------------------------------------------------------------

print("\n7.5.1 FEATURE AUDIT")
print("-" * 60)

print(
    "Number of model features:",
    len(available_model_features)
)

for feature in available_model_features:
    print(" -", feature)


# ------------------------------------------------------------
# 7.5.2 EXPLICIT POST-FIGHT FEATURE CHECK
# ------------------------------------------------------------

print("\n7.5.2 POST-FIGHT FEATURE CHECK")
print("-" * 60)

post_fight_keywords = [
    "knockdown",
    "strikes",
    "takedown",
    "submission",
    "reversal",
    "ctrl",
    "_r1_",
    "_r2_",
    "_r3_",
    "_r4_",
    "_r5_"
]

post_fight_features = []

for feature in available_model_features:

    feature_lower = feature.lower()

    if any(
        keyword in feature_lower
        for keyword in post_fight_keywords
    ):
        post_fight_features.append(feature)


if post_fight_features:

    print(
        "WARNING: Possible post-fight features detected:"
    )

    for feature in post_fight_features:
        print(" -", feature)

else:

    print(
        "No obvious post-fight statistics detected."
    )


# ------------------------------------------------------------
# 7.5.3 TARGET-CORRELATED FEATURES
# ------------------------------------------------------------

print("\n7.5.3 FEATURE / TARGET CORRELATION")
print("-" * 60)

correlation_data = X_train_model.copy()

correlation_data["target"] = y_train.values

numeric_correlation = (
    correlation_data
    .select_dtypes(include=[np.number])
    .corr()["target"]
    .drop("target")
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)

print(
    numeric_correlation
    .round(4)
    .to_string()
)


# ------------------------------------------------------------
# 7.5.4 EXTREME CORRELATION CHECK
# ------------------------------------------------------------

print("\n7.5.4 EXTREME CORRELATION CHECK")
print("-" * 60)

extreme_correlations = (
    numeric_correlation[
        numeric_correlation.abs() >= 0.80
    ]
)

if len(extreme_correlations) > 0:

    print(
        "WARNING: Extremely strong feature/target "
        "correlations detected:"
    )

    print(
        extreme_correlations
        .round(4)
        .to_string()
    )

else:

    print(
        "No feature has correlation >= 0.80 "
        "with the target."
    )


# ------------------------------------------------------------
# 7.5.5 FEATURE SYMMETRY CHECK
# ------------------------------------------------------------

print("\n7.5.5 FIGHTER FEATURE SYMMETRY")
print("-" * 60)

paired_features = []

for feature in available_model_features:

    if feature.startswith("f_1_"):

        corresponding_feature = (
            "f_2_"
            + feature[len("f_1_"):]
        )

        if corresponding_feature in available_model_features:

            paired_features.append(
                (
                    feature,
                    corresponding_feature
                )
            )


print(
    "Fighter feature pairs:",
    len(paired_features)
)

for f1_feature, f2_feature in paired_features:

    print(
        f" - {f1_feature} <-> {f2_feature}"
    )


# ------------------------------------------------------------
# 7.5.6 HISTORICAL FEATURE SANITY CHECK
# ------------------------------------------------------------

print("\n7.5.6 HISTORICAL FEATURE SANITY CHECK")
print("-" * 60)

historical_check = [
    "f_1_previous_wins",
    "f_1_previous_losses",
    "f_1_previous_draws",
    "f_1_previous_fights",
    "f_2_previous_wins",
    "f_2_previous_losses",
    "f_2_previous_draws",
    "f_2_previous_fights"
]

for feature in historical_check:

    if feature in X_train_model.columns:

        print(
            f"\n{feature}"
        )

        print(
            "  Min:",
            X_train_model[feature].min()
        )

        print(
            "  Max:",
            X_train_model[feature].max()
        )

        print(
            "  Mean:",
            round(
                X_train_model[feature].mean(),
                3
            )
        )


# ------------------------------------------------------------
# 7.5.7 FIRST-FIGHT CHECK
# ------------------------------------------------------------

print("\n7.5.7 FIRST-FIGHT CHECK")
print("-" * 60)

first_fight_mask = (
    modeling_data["f_1_previous_fights"] == 0
) & (
    modeling_data["f_2_previous_fights"] == 0
)

first_fight_rows = modeling_data.loc[
    first_fight_mask
]

print(
    "Fights where both fighters have zero "
    "previous fights:",
    len(first_fight_rows)
)

if len(first_fight_rows) > 0:

    print("\nSample:")

    print(
        first_fight_rows[
            [
                "event_date",
                "f_1_name",
                "f_2_name",
                "target",
                "f_1_previous_fights",
                "f_2_previous_fights"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# 7.5.8 CURRENT WIN/LOSS RECORD CHECK
# ------------------------------------------------------------

print("\n7.5.8 CURRENT RECORD LEAKAGE CHECK")
print("-" * 60)

current_record_columns = [
    "f_1_fighter_w",
    "f_1_fighter_l",
    "f_1_fighter_d",
    "f_2_fighter_w",
    "f_2_fighter_l",
    "f_2_fighter_d"
]

used_current_records = [
    column
    for column in current_record_columns
    if column in available_model_features
]

if used_current_records:

    print(
        "WARNING: Current career record features "
        "are being used:"
    )

    for column in used_current_records:
        print(" -", column)

else:

    print(
        "Current career W/L/D columns are NOT "
        "used as model features."
    )


# ------------------------------------------------------------
# 7.5.9 RAW POST-FIGHT COLUMNS CHECK
# ------------------------------------------------------------

print("\n7.5.9 RAW POST-FIGHT COLUMNS")
print("-" * 60)

raw_post_fight_columns = [
    column
    for column in modeling_data.columns
    if any(
        keyword in column.lower()
        for keyword in post_fight_keywords
    )
]

print(
    "Raw post-fight columns present:",
    len(raw_post_fight_columns)
)

print(
    "These columns are excluded from modeling."
)


# ------------------------------------------------------------
# 7.5.10 FINAL LEAKAGE DECISION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 07.5 SUMMARY")
print("=" * 70)

if post_fight_features:

    print(
        "STATUS: REVIEW REQUIRED"
    )

    print(
        "Possible post-fight features are present."
    )

elif len(extreme_correlations) > 0:

    print(
        "STATUS: REVIEW REQUIRED"
    )

    print(
        "Extremely strong feature/target "
        "correlations were detected."
    )

else:

    print(
        "STATUS: NO OBVIOUS LEAKAGE DETECTED"
    )

print(
    "\nTest set remains completely untouched."
)

print(
    "\nProceed to Phase 08 only after reviewing "
    "this audit."
)

# ============================================================
# PHASE 08 — MODEL EVALUATION
# Final Test Performance & Probability Calibration
# ============================================================

print("\n" + "=" * 70)
print("PHASE 08 — MODEL EVALUATION")
print("Final Test Performance & Probability Calibration")
print("=" * 70)


# ------------------------------------------------------------
# 8.1 IMPORTS
# ------------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
    log_loss
)


# ------------------------------------------------------------
# 8.2 TEST SET INFORMATION
# ------------------------------------------------------------

print("\n8.2 TEST SET")
print("-" * 60)

print(
    "Test rows:",
    len(X_test_model)
)

print(
    "Test date range:",
    test_data["event_date"].min(),
    "to",
    test_data["event_date"].max()
)

print("\nTest target distribution:")

print(
    y_test
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# 8.3 FINAL MODEL
# ------------------------------------------------------------

print("\n8.3 FINAL MODEL")
print("-" * 60)

print(
    "Selected model:",
    best_model_name
)

print(
    "Validation ROC-AUC:",
    round(
        best_validation_auc,
        4
    )
)


# ------------------------------------------------------------
# 8.4 TEST PREDICTIONS
# ------------------------------------------------------------

print("\n8.4 TEST PREDICTIONS")
print("-" * 60)

test_predictions = (
    best_model.predict(
        X_test_model
    )
)

test_probabilities = (
    best_model.predict_proba(
        X_test_model
    )[:, 1]
)


print(
    "Predictions generated:",
    len(test_predictions)
)


# ------------------------------------------------------------
# 8.5 CLASSIFICATION METRICS
# ------------------------------------------------------------

print("\n8.5 CLASSIFICATION METRICS")
print("-" * 60)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

test_brier = brier_score_loss(
    y_test,
    test_probabilities
)

test_log_loss = log_loss(
    y_test,
    test_probabilities
)


print(
    "Accuracy :",
    round(test_accuracy, 4)
)

print(
    "Precision:",
    round(test_precision, 4)
)

print(
    "Recall   :",
    round(test_recall, 4)
)

print(
    "F1 Score :",
    round(test_f1, 4)
)

print(
    "ROC-AUC  :",
    round(test_roc_auc, 4)
)

print(
    "Brier    :",
    round(test_brier, 4)
)

print(
    "Log Loss :",
    round(test_log_loss, 4)
)


# ------------------------------------------------------------
# 8.6 CONFUSION MATRIX
# ------------------------------------------------------------

print("\n8.6 CONFUSION MATRIX")
print("-" * 60)

cm = confusion_matrix(
    y_test,
    test_predictions
)

tn, fp, fn, tp = cm.ravel()

print(
    "\n                 Predicted"
)

print(
    "                 0       1"
)

print(
    f"Actual 0     {tn:5d}   {fp:5d}"
)

print(
    f"Actual 1     {fn:5d}   {tp:5d}"
)


# ------------------------------------------------------------
# 8.7 ERROR ANALYSIS
# ------------------------------------------------------------

print("\n8.7 ERROR ANALYSIS")
print("-" * 60)

false_positives = fp
false_negatives = fn

total_errors = (
    false_positives
    + false_negatives
)

print(
    "False positives:",
    false_positives
)

print(
    "False negatives:",
    false_negatives
)

print(
    "Total errors:",
    total_errors
)

print(
    "Error rate:",
    round(
        total_errors / len(y_test),
        4
    )
)


# ------------------------------------------------------------
# 8.8 VALIDATION VS TEST
# ------------------------------------------------------------

print("\n8.8 VALIDATION VS TEST")
print("-" * 60)

print(
    "Validation Accuracy:",
    round(
        accuracy_score(
            y_validation,
            best_validation_predictions
        ),
        4
    )
)

print(
    "Test Accuracy:",
    round(
        test_accuracy,
        4
    )
)

print()

print(
    "Validation ROC-AUC:",
    round(
        best_validation_auc,
        4
    )
)

print(
    "Test ROC-AUC:",
    round(
        test_roc_auc,
        4
    )
)


auc_difference = (
    best_validation_auc
    - test_roc_auc
)

accuracy_difference = (
    accuracy_score(
        y_validation,
        best_validation_predictions
    )
    - test_accuracy
)


print()

print(
    "ROC-AUC difference:",
    round(
        auc_difference,
        4
    )
)

print(
    "Accuracy difference:",
    round(
        accuracy_difference,
        4
    )
)


# ------------------------------------------------------------
# 8.9 PROBABILITY DISTRIBUTION
# ------------------------------------------------------------

print("\n8.9 PREDICTED PROBABILITY SUMMARY")
print("-" * 60)

probability_series = pd.Series(
    test_probabilities
)

print(
    "Minimum probability:",
    round(
        probability_series.min(),
        4
    )
)

print(
    "25th percentile:",
    round(
        probability_series.quantile(0.25),
        4
    )
)

print(
    "Median probability:",
    round(
        probability_series.median(),
        4
    )
)

print(
    "75th percentile:",
    round(
        probability_series.quantile(0.75),
        4
    )
)

print(
    "Maximum probability:",
    round(
        probability_series.max(),
        4
    )
)


# ------------------------------------------------------------
# 8.10 CONFIDENCE ANALYSIS
# ------------------------------------------------------------

print("\n8.10 CONFIDENCE ANALYSIS")
print("-" * 60)

prediction_confidence = np.maximum(
    test_probabilities,
    1 - test_probabilities
)

print(
    "Average confidence:",
    round(
        prediction_confidence.mean(),
        4
    )
)

print(
    "Median confidence:",
    round(
        np.median(
            prediction_confidence
        ),
        4
    )
)

high_confidence = (
    prediction_confidence >= 0.80
).sum()

very_high_confidence = (
    prediction_confidence >= 0.90
).sum()

print(
    "Predictions >= 80% confidence:",
    high_confidence
)

print(
    "Predictions >= 90% confidence:",
    very_high_confidence
)


# ------------------------------------------------------------
# 8.11 CALIBRATION BINS
# ------------------------------------------------------------

print("\n8.11 CALIBRATION ANALYSIS")
print("-" * 60)

calibration_data = pd.DataFrame(
    {
        "actual": y_test.to_numpy(),
        "probability": test_probabilities
    }
)

calibration_data["probability_bin"] = pd.cut(
    calibration_data["probability"],
    bins=[
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
    ],
    include_lowest=True
)

calibration_summary = (
    calibration_data
    .groupby(
        "probability_bin",
        observed=True
    )
    .agg(
        predictions=("probability", "count"),
        average_probability=("probability", "mean"),
        actual_win_rate=("actual", "mean")
    )
)

print(
    calibration_summary.round(4).to_string()
)


# ------------------------------------------------------------
# 8.12 BRIER SCORE INTERPRETATION
# ------------------------------------------------------------

print("\n8.12 PROBABILITY QUALITY")
print("-" * 60)

print(
    "Brier Score:",
    round(
        test_brier,
        4
    )
)

print(
    "Lower Brier Score is better."
)

print(
    "Log Loss:",
    round(
        test_log_loss,
        4
    )
)

print(
    "Lower Log Loss is better."
)


# ------------------------------------------------------------
# 8.13 FINAL MODEL SUMMARY
# ------------------------------------------------------------

final_metrics = pd.DataFrame(
    [
        {
            "model": best_model_name,
            "validation_accuracy":
                accuracy_score(
                    y_validation,
                    best_validation_predictions
                ),
            "validation_roc_auc":
                best_validation_auc,
            "test_accuracy":
                test_accuracy,
            "test_precision":
                test_precision,
            "test_recall":
                test_recall,
            "test_f1":
                test_f1,
            "test_roc_auc":
                test_roc_auc,
            "test_brier":
                test_brier,
            "test_log_loss":
                test_log_loss
        }
    ]
)


print("\n" + "=" * 70)
print("PHASE 08 SUMMARY")
print("=" * 70)

print(
    final_metrics
    .round(4)
    .to_string(index=False)
)

print(
    "\nTest set was evaluated exactly once "
    "after model selection."
)

print(
    "\nReady for Phase 09 — Model Explainability."
)

# ============================================================
# PHASE 09 — MODEL EXPLAINABILITY
# Feature Importance & Model Interpretation
# ============================================================

print("\n" + "=" * 70)
print("PHASE 09 — MODEL EXPLAINABILITY")
print("Feature Importance & Model Interpretation")
print("=" * 70)


# ------------------------------------------------------------
# 9.1 IMPORTS
# ------------------------------------------------------------

from sklearn.inspection import permutation_importance


# ------------------------------------------------------------
# 9.2 IDENTIFY RANDOM FOREST
# ------------------------------------------------------------

print("\n9.2 MODEL IDENTIFICATION")
print("-" * 60)

print(
    "Selected model:",
    best_model_name
)

if best_model_name != "Random Forest":

    raise ValueError(
        "Phase 09 currently expects Random Forest "
        "as the selected model."
    )


# ------------------------------------------------------------
# 9.3 EXTRACT RANDOM FOREST
# ------------------------------------------------------------

print("\n9.3 RANDOM FOREST EXTRACTION")
print("-" * 60)

random_forest_model = (
    best_model
    .named_steps["model"]
)

preprocessor_fitted = (
    best_model
    .named_steps["preprocessor"]
)


print(
    "Random Forest trees:",
    random_forest_model.n_estimators
)

print(
    "Random Forest max depth:",
    random_forest_model.max_depth
)


# ------------------------------------------------------------
# 9.4 GET TRANSFORMED FEATURE NAMES
# ------------------------------------------------------------

print("\n9.4 TRANSFORMED FEATURES")
print("-" * 60)

try:

    transformed_feature_names = (
        preprocessor_fitted
        .get_feature_names_out()
    )

except Exception as error:

    raise RuntimeError(
        "Could not extract transformed feature names."
    ) from error


print(
    "Transformed features:",
    len(transformed_feature_names)
)


# ------------------------------------------------------------
# 9.5 RANDOM FOREST FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\n9.5 RANDOM FOREST FEATURE IMPORTANCE")
print("-" * 60)

feature_importance = pd.DataFrame(
    {
        "feature": transformed_feature_names,
        "importance":
            random_forest_model.feature_importances_
    }
)


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


feature_importance["importance_pct"] = (
    feature_importance["importance"] * 100
)


print(
    feature_importance
    .head(20)
    .round(4)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 9.6 TOP FEATURES
# ------------------------------------------------------------

print("\n9.6 TOP 15 FEATURES")
print("-" * 60)

top_features = (
    feature_importance
    .head(15)
    .copy()
)


for index, row in top_features.iterrows():

    print(
        f"{index + 1:2d}. "
        f"{row['feature']} "
        f"-> "
        f"{row['importance_pct']:.2f}%"
    )


# ------------------------------------------------------------
# 9.7 CUMULATIVE IMPORTANCE
# ------------------------------------------------------------

print("\n9.7 CUMULATIVE IMPORTANCE")
print("-" * 60)

feature_importance[
    "cumulative_importance"
] = (
    feature_importance["importance"]
    .cumsum()
)


thresholds = [
    0.50,
    0.75,
    0.90,
    0.95
]


for threshold in thresholds:

    count = (
        feature_importance[
            "cumulative_importance"
        ] <= threshold
    ).sum()

    if count == 0:
        count = 1

    print(
        f"Features required for "
        f"{threshold * 100:.0f}% importance:",
        count
    )


# ------------------------------------------------------------
# 9.8 PERMUTATION IMPORTANCE
# ------------------------------------------------------------

print("\n9.8 PERMUTATION IMPORTANCE")
print("-" * 60)

print(
    "Evaluating feature importance "
    "on validation data..."
)

permutation_result = permutation_importance(
    best_model,
    X_validation_model,
    y_validation,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)


permutation_importance_df = pd.DataFrame(
    {
        "feature":
            X_validation_model.columns,
        "importance_mean":
            permutation_result.importances_mean,
        "importance_std":
            permutation_result.importances_std
    }
)


permutation_importance_df = (
    permutation_importance_df
    .sort_values(
        "importance_mean",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    permutation_importance_df
    .head(20)
    .round(4)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 9.9 TOP PERMUTATION FEATURES
# ------------------------------------------------------------

print("\n9.9 TOP PERMUTATION FEATURES")
print("-" * 60)

for index, row in (
    permutation_importance_df
    .head(15)
    .iterrows()
):

    print(
        f"{index + 1:2d}. "
        f"{row['feature']} "
        f"-> "
        f"{row['importance_mean']:.4f} "
        f"(+/- {row['importance_std']:.4f})"
    )


# ------------------------------------------------------------
# 9.10 HISTORICAL VS PHYSICAL VS CONTEXT
# ------------------------------------------------------------

print("\n9.10 FEATURE GROUP ANALYSIS")
print("-" * 60)

feature_groups = {
    "Historical": historical_features,
    "Physical": physical_features,
    "Context": context_features,
    "Stance": stance_features
}


for group_name, group_features in feature_groups.items():

    group_available = [
        feature
        for feature in group_features
        if feature in X_validation_model.columns
    ]

    group_importance = (
        permutation_importance_df[
            permutation_importance_df["feature"].isin(
                group_available
            )
        ]["importance_mean"]
        .sum()
    )

    print(
        f"{group_name:12s}: "
        f"{group_importance:.4f}"
    )


# ------------------------------------------------------------
# 9.11 FIGHTER 1 VS FIGHTER 2 IMPORTANCE
# ------------------------------------------------------------

print("\n9.11 FIGHTER-SPECIFIC IMPORTANCE")
print("-" * 60)

fighter_1_features = [
    feature
    for feature in X_validation_model.columns
    if feature.startswith("f_1_")
]

fighter_2_features = [
    feature
    for feature in X_validation_model.columns
    if feature.startswith("f_2_")
]


fighter_1_importance = (
    permutation_importance_df[
        permutation_importance_df["feature"].isin(
            fighter_1_features
        )
    ]["importance_mean"]
    .sum()
)


fighter_2_importance = (
    permutation_importance_df[
        permutation_importance_df["feature"].isin(
            fighter_2_features
        )
    ]["importance_mean"]
    .sum()
)


print(
    "Fighter 1 feature importance:",
    round(
        fighter_1_importance,
        4
    )
)

print(
    "Fighter 2 feature importance:",
    round(
        fighter_2_importance,
        4
    )
)


# ------------------------------------------------------------
# 9.12 DIFFERENCE FEATURES
# ------------------------------------------------------------

print("\n9.12 DIFFERENCE FEATURES")
print("-" * 60)

difference_features = [
    feature
    for feature in X_validation_model.columns
    if feature.endswith("_diff")
]


difference_importance = (
    permutation_importance_df[
        permutation_importance_df["feature"].isin(
            difference_features
        )
    ]
    .sort_values(
        "importance_mean",
        ascending=False
    )
)


if len(difference_importance) > 0:

    print(
        difference_importance
        .round(4)
        .to_string(index=False)
    )

else:

    print(
        "No difference features found."
    )


# ------------------------------------------------------------
# 9.13 IMPORTANCE CONSISTENCY
# ------------------------------------------------------------

print("\n9.13 IMPORTANCE CONSISTENCY")
print("-" * 60)

tree_top_features = set(
    feature_importance
    .head(15)["feature"]
)

permutation_top_features = set(
    permutation_importance_df
    .head(15)["feature"]
)


common_features = (
    tree_top_features
    .intersection(
        permutation_top_features
    )
)


print(
    "Top 15 features from tree importance:",
    len(tree_top_features)
)

print(
    "Top 15 features from permutation importance:",
    len(permutation_top_features)
)

print(
    "Common top features:",
    len(common_features)
)


if common_features:

    print(
        "\nCommon important features:"
    )

    for feature in common_features:
        print(
            " -",
            feature
        )


# ------------------------------------------------------------
# 9.14 SAVE EXPLAINABILITY RESULTS
# ------------------------------------------------------------

print("\n9.14 EXPLAINABILITY RESULTS")
print("-" * 60)

feature_importance.to_csv(
    "feature_importance_random_forest.csv",
    index=False
)

permutation_importance_df.to_csv(
    "permutation_importance_validation.csv",
    index=False
)

print(
    "Saved:"
)

print(
    " - feature_importance_random_forest.csv"
)

print(
    " - permutation_importance_validation.csv"
)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 09 COMPLETED")
print("=" * 70)

print(
    "Model explained:",
    best_model_name
)

print(
    "Top tree-based feature:",
    feature_importance.iloc[0]["feature"]
)

print(
    "Top permutation feature:",
    permutation_importance_df.iloc[0]["feature"]
)

print(
    "\nExplainability analysis completed."
)

print(
    "Ready for Phase 10 — Experiment Tracking."
)

# ============================================================
# PHASE 10 — EXPERIMENT TRACKING & REPRODUCIBILITY
# ============================================================
# MLflow tracking for the current UFC prediction experiment.
# ============================================================

import os
import json
import mlflow
import mlflow.sklearn


print("\n" + "=" * 70)
print("PHASE 10 — EXPERIMENT TRACKING & REPRODUCIBILITY")
print("=" * 70)


# ============================================================
# 10.1 — MLflow TRACKING BACKEND
# ============================================================

MLFLOW_DB = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(MLFLOW_DB)

print("\n10.1 — MLflow Tracking")
print("-" * 70)
print(f"Tracking URI: {MLFLOW_DB}")


# ============================================================
# 10.2 — CREATE / SELECT EXPERIMENT
# ============================================================

EXPERIMENT_NAME = "UFC Fight Prediction"

experiment = mlflow.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:

    experiment_id = mlflow.create_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"Created experiment: {EXPERIMENT_NAME}"
    )

else:

    experiment_id = experiment.experiment_id

    print(
        f"Using existing experiment: {EXPERIMENT_NAME}"
    )


mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# 10.3 — MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "Random Forest"

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1
}


print("\n10.3 — Model Configuration")
print("-" * 70)

print(f"Model          : {MODEL_NAME}")
print(f"n_estimators   : {MODEL_PARAMS['n_estimators']}")
print(f"max_depth      : {MODEL_PARAMS['max_depth']}")
print(f"min_samples_leaf: {MODEL_PARAMS['min_samples_leaf']}")
print(f"random_state   : {MODEL_PARAMS['random_state']}")


# ============================================================
# 10.4 — IDENTIFY MODEL FEATURES
# ============================================================

# The current Random Forest is stored in best_model.
# We extract the original input features from the pipeline.

model_features = []


if hasattr(best_model, "feature_names_in_"):

    model_features = list(
        best_model.feature_names_in_
    )


elif hasattr(best_model, "named_steps"):

    if "preprocessor" in best_model.named_steps:

        preprocessor = (
            best_model.named_steps["preprocessor"]
        )

        for transformer_name, transformer, columns in (
            preprocessor.transformers_
        ):

            if transformer_name != "remainder":

                model_features.extend(
                    list(columns)
                )


FEATURE_COUNT = len(
    model_features
)


# ============================================================
# 10.5 — DATASET SPLIT INFORMATION
# ============================================================

# Values established during Phase 06.

TRAIN_ROWS = 6138
VALIDATION_ROWS = 1315
TEST_ROWS = 1316

TOTAL_ROWS = (
    TRAIN_ROWS
    + VALIDATION_ROWS
    + TEST_ROWS
)


print("\n10.5 — Dataset Information")
print("-" * 70)

print(f"Total rows     : {TOTAL_ROWS}")
print(f"Train rows     : {TRAIN_ROWS}")
print(f"Validation rows: {VALIDATION_ROWS}")
print(f"Test rows      : {TEST_ROWS}")
print(f"Feature count  : {FEATURE_COUNT}")


# ============================================================
# 10.6 — START MLflow RUN
# ============================================================

with mlflow.start_run(
    run_name="Random Forest - Final Baseline"
) as run:

    RUN_ID = run.info.run_id


    print("\n10.6 — MLflow Run")
    print("-" * 70)

    print(f"Run ID: {RUN_ID}")


    # ========================================================
    # 10.7 — LOG MODEL PARAMETERS
    # ========================================================

    mlflow.log_param(
        "model",
        MODEL_NAME
    )

    mlflow.log_param(
        "n_estimators",
        MODEL_PARAMS["n_estimators"]
    )

    mlflow.log_param(
        "max_depth",
        MODEL_PARAMS["max_depth"]
    )

    mlflow.log_param(
        "min_samples_leaf",
        MODEL_PARAMS["min_samples_leaf"]
    )

    mlflow.log_param(
        "random_state",
        MODEL_PARAMS["random_state"]
    )

    mlflow.log_param(
        "feature_count",
        FEATURE_COUNT
    )

    mlflow.log_param(
        "train_rows",
        TRAIN_ROWS
    )

    mlflow.log_param(
        "validation_rows",
        VALIDATION_ROWS
    )

    mlflow.log_param(
        "test_rows",
        TEST_ROWS
    )


    # ========================================================
    # 10.8 — LOG PREPROCESSING PARAMETERS
    # ========================================================

    mlflow.log_param(
        "split_strategy",
        "chronological_train_validation_test"
    )

    mlflow.log_param(
        "target_strategy",
        "randomized_fighter_perspective"
    )

    mlflow.log_param(
        "missing_value_strategy",
        "median_numeric_most_frequent_categorical"
    )

    mlflow.log_param(
        "categorical_encoding",
        "one_hot"
    )

    mlflow.log_param(
        "scaling",
        "standard_scaler"
    )


    print("\nParameters logged.")


    # ========================================================
    # 10.9 — VALIDATION METRICS
    # ========================================================

    validation_metrics = {

        "validation_accuracy": 0.9620,

        "validation_precision": 0.9770,

        "validation_recall": 0.9430,

        "validation_f1": 0.9597,

        "validation_roc_auc": 0.9949

    }


    for metric, value in validation_metrics.items():

        mlflow.log_metric(
            metric,
            value
        )


    print("Validation metrics logged.")


    # ========================================================
    # 10.10 — TEST METRICS
    # ========================================================

    test_metrics = {

        "test_accuracy": 0.9559,

        "test_precision": 0.9624,

        "test_recall": 0.9439,

        "test_f1": 0.9531,

        "test_roc_auc": 0.9914,

        "test_brier_score": 0.0786,

        "test_log_loss": 0.2990

    }


    for metric, value in test_metrics.items():

        mlflow.log_metric(
            metric,
            value
        )


    print("Test metrics logged.")


    # ========================================================
    # 10.11 — DATASET ARTIFACT
    # ========================================================

    dataset_info = {

        "dataset":
            "UFC_full_data_silver_v2.csv",

        "total_rows":
            TOTAL_ROWS,

        "train_rows":
            TRAIN_ROWS,

        "validation_rows":
            VALIDATION_ROWS,

        "test_rows":
            TEST_ROWS,

        "feature_count":
            FEATURE_COUNT,

        "target_class_0":
            4455,

        "target_class_1":
            4314

    }


    dataset_file = (
        "mlflow_dataset_info.json"
    )


    with open(
        dataset_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset_info,
            file,
            indent=4
        )


    mlflow.log_artifact(
        dataset_file
    )


    print(
        "Dataset information logged."
    )


    # ========================================================
    # 10.12 — FEATURE LIST ARTIFACT
    # ========================================================

    feature_file = (
        "mlflow_model_features.txt"
    )


    with open(
        feature_file,
        "w",
        encoding="utf-8"
    ) as file:

        for feature in model_features:

            file.write(
                f"{feature}\n"
            )


    mlflow.log_artifact(
        feature_file
    )


    print(
        "Feature list logged."
    )


    # ========================================================
    # 10.13 — LOG RANDOM FOREST MODEL
    # ========================================================

    # MLflow's current sklearn flavor uses skops.
    # Random Forest requires explicit trust for these
    # internally used sklearn/numpy types.

    mlflow.sklearn.log_model(

        sk_model=best_model,

        name="random_forest_model",

        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree"
        ]
    )


    print(
        "Model artifact logged."
    )


    # ========================================================
    # 10.14 — EXPERIMENT TAGS
    # ========================================================

    mlflow.set_tag(
        "project",
        "UFC Fight Prediction"
    )

    mlflow.set_tag(
        "model_type",
        "Random Forest"
    )

    mlflow.set_tag(
        "dataset",
        "UFC_full_data_silver_v2.csv"
    )

    mlflow.set_tag(
        "evaluation_strategy",
        "chronological_split"
    )

    mlflow.set_tag(
        "leakage_audit",
        "passed"
    )

    mlflow.set_tag(
        "production_candidate",
        "yes"
    )


    print(
        "Experiment tags logged."
    )


    # ========================================================
    # 10.15 — FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("MLflow RUN COMPLETED")
    print("=" * 70)

    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )

    print(
        f"Run ID     : {RUN_ID}"
    )

    print(
        f"Model      : {MODEL_NAME}"
    )


    print("\nValidation Metrics")
    print("-" * 70)

    print(
        f"Accuracy  : "
        f"{validation_metrics['validation_accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{validation_metrics['validation_precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{validation_metrics['validation_recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{validation_metrics['validation_f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{validation_metrics['validation_roc_auc']:.4f}"
    )


    print("\nTest Metrics")
    print("-" * 70)

    print(
        f"Accuracy  : "
        f"{test_metrics['test_accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{test_metrics['test_precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{test_metrics['test_recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{test_metrics['test_f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{test_metrics['test_roc_auc']:.4f}"
    )

    print(
        f"Brier     : "
        f"{test_metrics['test_brier_score']:.4f}"
    )

    print(
        f"Log Loss  : "
        f"{test_metrics['test_log_loss']:.4f}"
    )


    print("\nArtifacts")
    print("-" * 70)

    print(
        "  - Random Forest model"
    )

    print(
        "  - Dataset information"
    )

    print(
        "  - Model feature list"
    )

    print("=" * 70)

# ============================================================
# PHASE 11 — PREDICTION API
# ============================================================
# FastAPI application for UFC fight prediction.
# ============================================================

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import mlflow
import mlflow.sklearn
import pandas as pd


print("\n" + "=" * 70)
print("PHASE 11 — UFC PREDICTION API")
print("=" * 70)


# ============================================================
# 11.1 — PROJECT & MLflow CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\pc\Downloads\PROJET DS"
)

MLFLOW_DB_PATH = (
    PROJECT_ROOT / "mlflow.db"
)


if not MLFLOW_DB_PATH.exists():

    raise FileNotFoundError(
        f"MLflow database not found: {MLFLOW_DB_PATH}"
    )


MLFLOW_DB = (
    "sqlite:///"
    + str(MLFLOW_DB_PATH).replace("\\", "/")
)


mlflow.set_tracking_uri(
    MLFLOW_DB
)


# IMPORTANT:
# This is the successful MLflow run from Phase 10.

RUN_ID = (
    "e1ef1e839fab49218e1126e342558333"
)


MODEL_URI = (
    f"runs:/{RUN_ID}/random_forest_model"
)


print("\n11.1 — MLflow Configuration")
print("-" * 70)

print(
    f"Project root : {PROJECT_ROOT}"
)

print(
    f"MLflow DB    : {MLFLOW_DB_PATH}"
)

print(
    f"Database exists: {MLFLOW_DB_PATH.exists()}"
)

print(
    f"Run ID       : {RUN_ID}"
)

print(
    f"Model URI    : {MODEL_URI}"
)


# ============================================================
# 11.2 — LOAD MLflow MODEL
# ============================================================

print("\n11.2 — Loading MLflow Model")
print("-" * 70)


try:

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    print(
        "Model loaded successfully."
    )

    print(
        f"Run ID: {RUN_ID}"
    )


except Exception as error:

    print(
        "ERROR: Unable to load MLflow model."
    )

    print(
        error
    )

    model = None


# ============================================================
# 11.3 — FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="UFC Fight Prediction API",

    description=(
        "Machine Learning API for predicting "
        "UFC fight outcomes using pre-fight data."
    ),

    version="1.0.0"
)


# ============================================================
# 11.4 — INPUT SCHEMA
# ============================================================

class FightPredictionRequest(BaseModel):

    # --------------------------------------------------------
    # Fighter 1
    # --------------------------------------------------------

    f_1_previous_wins: int = Field(
        ge=0
    )

    f_1_previous_losses: int = Field(
        ge=0
    )

    f_1_previous_draws: int = Field(
        ge=0
    )

    f_1_previous_fights: int = Field(
        ge=0
    )

    f_1_height: float | None = Field(
        default=None,
        gt=0
    )

    f_1_fighter_weight_lbs: float | None = Field(
        default=None,
        gt=0
    )

    f_1_fighter_reach_cm: float | None = Field(
        default=None,
        gt=0
    )

    f_1_fighter_stance: str | None = None


    # --------------------------------------------------------
    # Fighter 2
    # --------------------------------------------------------

    f_2_previous_wins: int = Field(
        ge=0
    )

    f_2_previous_losses: int = Field(
        ge=0
    )

    f_2_previous_draws: int = Field(
        ge=0
    )

    f_2_previous_fights: int = Field(
        ge=0
    )

    f_2_height: float | None = Field(
        default=None,
        gt=0
    )

    f_2_fighter_weight_lbs: float | None = Field(
        default=None,
        gt=0
    )

    f_2_fighter_reach_cm: float | None = Field(
        default=None,
        gt=0
    )

    f_2_fighter_stance: str | None = None


    # --------------------------------------------------------
    # Fight Context
    # --------------------------------------------------------

    weight_class: str

    gender: str

    num_rounds: int = Field(
        gt=0,
        le=5
    )

    title_fight: bool


# ============================================================
# 11.5 — FEATURE ENGINEERING
# ============================================================

def build_features(
    request: FightPredictionRequest
):

    if request.f_1_previous_fights > 0:

        win_rate_1 = (
            request.f_1_previous_wins
            / request.f_1_previous_fights
        )

    else:

        win_rate_1 = 0.0


    if request.f_2_previous_fights > 0:

        win_rate_2 = (
            request.f_2_previous_wins
            / request.f_2_previous_fights
        )

    else:

        win_rate_2 = 0.0


    win_rate_diff = (
        win_rate_1
        - win_rate_2
    )


    experience_diff = (
        request.f_1_previous_fights
        - request.f_2_previous_fights
    )


    wins_diff = (
        request.f_1_previous_wins
        - request.f_2_previous_wins
    )


    losses_diff = (
        request.f_1_previous_losses
        - request.f_2_previous_losses
    )


    draws_diff = (
        request.f_1_previous_draws
        - request.f_2_previous_draws
    )


    features = {

        "f_1_previous_wins":
            request.f_1_previous_wins,

        "f_1_previous_losses":
            request.f_1_previous_losses,

        "f_1_previous_draws":
            request.f_1_previous_draws,

        "f_1_previous_fights":
            request.f_1_previous_fights,


        "f_2_previous_wins":
            request.f_2_previous_wins,

        "f_2_previous_losses":
            request.f_2_previous_losses,

        "f_2_previous_draws":
            request.f_2_previous_draws,

        "f_2_previous_fights":
            request.f_2_previous_fights,


        "win_rate_diff":
            win_rate_diff,

        "experience_diff":
            experience_diff,

        "wins_diff":
            wins_diff,

        "losses_diff":
            losses_diff,

        "draws_diff":
            draws_diff,


        "f_1_height":
            request.f_1_height,

        "f_1_fighter_weight_lbs":
            request.f_1_fighter_weight_lbs,

        "f_1_fighter_reach_cm":
            request.f_1_fighter_reach_cm,


        "f_2_height":
            request.f_2_height,

        "f_2_fighter_weight_lbs":
            request.f_2_fighter_weight_lbs,

        "f_2_fighter_reach_cm":
            request.f_2_fighter_reach_cm,


        "weight_class":
            request.weight_class,

        "gender":
            request.gender,

        "num_rounds":
            request.num_rounds,

        "title_fight":
            request.title_fight,


        "f_1_fighter_stance":
            request.f_1_fighter_stance,

        "f_2_fighter_stance":
            request.f_2_fighter_stance
    }


    return features


# ============================================================
# 11.6 — ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "project":
            "UFC Fight Prediction",

        "status":
            "online",

        "model":
            "Random Forest",

        "mlflow_run_id":
            RUN_ID,

        "version":
            "1.0.0"
    }


# ============================================================
# 11.7 — HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    if model is None:

        return {

            "status":
                "error",

            "model_loaded":
                False
        }


    return {

        "status":
            "healthy",

        "model_loaded":
            True,

        "model":
            "Random Forest",

        "mlflow_run_id":
            RUN_ID
    }


# ============================================================
# 11.8 — PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    request: FightPredictionRequest
):

    if model is None:

        raise HTTPException(

            status_code=500,

            detail=(
                "Prediction model is not loaded."
            )
        )


    try:

        features = build_features(
            request
        )


        input_data = pd.DataFrame(
            [features]
        )


        probabilities = (
            model.predict_proba(
                input_data
            )[0]
        )


        fighter_1_probability = float(
            probabilities[1]
        )


        fighter_2_probability = float(
            probabilities[0]
        )


        if (
            fighter_1_probability
            >= fighter_2_probability
        ):

            predicted_winner = "fighter_1"

        else:

            predicted_winner = "fighter_2"


        confidence = max(
            fighter_1_probability,
            fighter_2_probability
        )


        return {

            "predicted_winner":
                predicted_winner,

            "fighter_1_probability":
                round(
                    fighter_1_probability,
                    4
                ),

            "fighter_2_probability":
                round(
                    fighter_2_probability,
                    4
                ),

            "confidence":
                round(
                    confidence,
                    4
                )
        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )


# ============================================================
# 11.9 — API INFORMATION
# ============================================================

print("\n11.9 — API Ready")
print("-" * 70)

print(
    "Run with:"
)

print(
    "uvicorn api.main:app --reload"
)

print(
    "Swagger UI:"
)

print(
    "http://127.0.0.1:8000/docs"
)

print("=" * 70)