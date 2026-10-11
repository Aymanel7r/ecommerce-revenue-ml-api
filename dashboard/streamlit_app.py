from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UFC_full_data_silver_v2.csv"
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="OctagonIQ | UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #101114;
        color: #F5F5F5;
    }

    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #292929, #111111);
        border: 1px solid #454545;
        margin-bottom: 25px;
    }

    .fighter-card {
        background: #1B1C20;
        border: 1px solid #393A40;
        border-radius: 16px;
        padding: 22px;
    }

    .winner-card {
        background: linear-gradient(135deg, #173525, #10251b);
        border: 1px solid #2C8C57;
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        margin: 20px 0;
    }

    .winner-label {
        color: #65E69A;
        font-size: 14px;
        letter-spacing: 3px;
        font-weight: bold;
    }

    .winner-name {
        color: white;
        font-size: 34px;
        font-weight: bold;
        margin: 10px 0;
    }

    .winner-probability {
        color: #65E69A;
        font-size: 42px;
        font-weight: bold;
    }

    div.stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    required = ["f_1_name", "f_2_name"]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    for column in required:
        df[column] = df[column].astype("string").str.strip()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()


# =========================================================
# FIGHTER DIRECTORY
# =========================================================

def build_fighter_directory(data):
    names = set()

    for column in ["f_1_name", "f_2_name"]:
        values = data[column].dropna().astype(str).str.strip()
        names.update(
            name for name in values
            if name and name.lower() != "nan"
        )

    return sorted(names, key=str.casefold)


fighter_names = build_fighter_directory(df)


# =========================================================
# FIGHTER PROFILE
# =========================================================

def get_fighter_profile(name):
    candidates = []

    profile_columns = {
        "f_1_name": {
            "height": "f_1_fighter_height_cm",
            "weight": "f_1_fighter_weight_lbs",
            "reach": "f_1_fighter_reach_cm",
            "stance": "f_1_fighter_stance",
        },
        "f_2_name": {
            "height": "f_2_fighter_height_cm",
            "weight": "f_2_fighter_weight_lbs",
            "reach": "f_2_fighter_reach_cm",
            "stance": "f_2_fighter_stance",
        },
    }

    for name_column, mapping in profile_columns.items():
        if name_column not in df.columns:
            continue

        matches = df[df[name_column] == name]

        for _, row in matches.iterrows():
            profile = {
                key: row.get(column)
                for key, column in mapping.items()
            }

            profile["weight_class"] = row.get("weight_class")
            profile["date"] = row.get("date")

            candidates.append(profile)

    if not candidates:
        return {}

    profiles = pd.DataFrame(candidates)

    if "date" in profiles.columns:
        profiles = profiles.sort_values("date", na_position="first")

    # Most recent recorded profile, not necessarily current measurements.
    return profiles.iloc[-1].to_dict()


def safe_number(value):
    try:
        number = float(value)
        if pd.isna(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def format_value(value, suffix=""):
    number = safe_number(value)

    if number is None:
        return "N/A"

    if number.is_integer():
        return f"{int(number)}{suffix}"

    return f"{number:.1f}{suffix}"


def format_text(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return "N/A"
    return str(value)


# =========================================================
# HISTORICAL RECORD
# =========================================================

def get_historical_record(name):
    """
    Calculate wins/losses/draws from fights before the latest
    recorded fight date in the dataset.

    This is an approximate profile record. For a specific upcoming
    fight date, records should instead be calculated as of that date.
    """

    if "winner" not in df.columns:
        return {"wins": 0, "losses": 0, "draws": 0, "fights": 0}

    wins = 0
    losses = 0
    draws = 0

    for _, row in df.iterrows():
        fighter_1 = row.get("f_1_name")
        fighter_2 = row.get("f_2_name")
        winner = row.get("winner")

        if name != fighter_1 and name != fighter_2:
            continue

        if pd.isna(winner):
            continue

        winner = str(winner).strip()

        if winner.lower() in {"draw", "draws", "nc", "no contest"}:
            draws += 1
        elif winner == name:
            wins += 1
        else:
            losses += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "fights": wins + losses + draws,
    }


# =========================================================
# API STATUS
# =========================================================

def api_is_available():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.ok
    except requests.RequestException:
        return False


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🥊 OctagonIQ</h1>
        <p>UFC Fight Prediction & Fighter Analytics</p>
        <p>Compare fighters, explore their profiles, and predict matchups.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if api_is_available():
    st.success("Prediction API connected")
else:
    st.warning("Prediction API is offline. Start FastAPI in another terminal.")

st.caption(f"Fighters available in dataset: {len(fighter_names):,}")


# =========================================================
# FIGHTER SELECTION
# =========================================================

st.header("1. Select the Matchup")

col_a, col_b = st.columns(2)

with col_a:
    fighter_a = st.selectbox(
        "🔴 Fighter A",
        fighter_names,
        key="fighter_a",
    )

with col_b:
    fighter_b_options = [name for name in fighter_names if name != fighter_a]

    fighter_b = st.selectbox(
        "🔵 Fighter B",
        fighter_b_options,
        key="fighter_b",
    )


profile_a = get_fighter_profile(fighter_a)
profile_b = get_fighter_profile(fighter_b)

record_a = get_historical_record(fighter_a)
record_b = get_historical_record(fighter_b)


# =========================================================
# FIGHTER CARDS
# =========================================================

st.header("2. Fighter Profiles")


def render_fighter_card(name, profile, record, key):
    st.subheader(name)

    uploaded_image = st.file_uploader(
        f"Upload photo for {name}",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"photo_{key}",
    )

    if uploaded_image is not None:
        st.image(uploaded_image, use_container_width=True)
    else:
        st.markdown(
            """
            <div style="
                height: 180px;
                display: flex;
                justify-content: center;
                align-items: center;
                background: #202126;
                border: 1px dashed #555;
                border-radius: 12px;
                margin-bottom: 15px;
                color: #AAAAAA;
            ">
                No photo uploaded
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)

    c1.metric("Height", format_value(profile.get("height"), " cm"))
    c2.metric("Weight", format_value(profile.get("weight"), " lbs"))

    c3, c4 = st.columns(2)

    c3.metric("Reach", format_value(profile.get("reach"), " cm"))
    c4.metric("Stance", format_text(profile.get("stance")))

    st.markdown("#### Historical record")

    r1, r2, r3 = st.columns(3)

    r1.metric("Wins", record["wins"])
    r2.metric("Losses", record["losses"])
    r3.metric("Draws", record["draws"])

    st.caption(
        "Record calculated from available dataset rows. "
        "Verify source data before interpreting as an official career record."
    )


col_a, col_b = st.columns(2)

with col_a:
    render_fighter_card(fighter_a, profile_a, record_a, "a")

with col_b:
    render_fighter_card(fighter_b, profile_b, record_b, "b")


# =========================================================
# TALE OF THE TAPE
# =========================================================

st.header("3. Tale of the Tape")

comparison = pd.DataFrame(
    {
        "Statistic": [
            "Height (cm)",
            "Weight (lbs)",
            "Reach (cm)",
            "Historical wins",
            "Historical losses",
            "Historical draws",
        ],
        fighter_a: [
            format_value(profile_a.get("height")),
            format_value(profile_a.get("weight")),
            format_value(profile_a.get("reach")),
            record_a["wins"],
            record_a["losses"],
            record_a["draws"],
        ],
        fighter_b: [
            format_value(profile_b.get("height")),
            format_value(profile_b.get("weight")),
            format_value(profile_b.get("reach")),
            record_b["wins"],
            record_b["losses"],
            record_b["draws"],
        ],
    }
)

st.dataframe(comparison, hide_index=True, use_container_width=True)


# =========================================================
# FIGHT SETTINGS
# =========================================================

st.header("4. Fight Configuration")

weight_classes = (
    sorted(df["weight_class"].dropna().astype(str).unique())
    if "weight_class" in df.columns
    else ["Unknown"]
)

genders = (
    sorted(df["gender"].dropna().astype(str).unique())
    if "gender" in df.columns
    else ["Unknown"]
)

settings_a, settings_b, settings_c = st.columns(3)

with settings_a:
    selected_weight_class = st.selectbox(
        "Weight class",
        weight_classes,
    )

with settings_b:
    num_rounds = st.selectbox(
        "Number of rounds",
        [3, 5],
    )

with settings_c:
    title_fight = st.checkbox("Title fight")


gender = st.selectbox("Gender", genders)


# =========================================================
# PREDICTION INPUTS
# =========================================================

st.header("5. Pre-Fight Inputs")

st.info(
    "The profiles above show historical dataset information. "
    "Check the values below and adjust them if needed for the upcoming fight."
)


def fighter_prediction_inputs(name, profile, record, key):
    st.subheader(name)

    c1, c2, c3 = st.columns(3)

    with c1:
        wins = st.number_input(
            "Previous wins",
            min_value=0,
            value=int(record["wins"]),
            key=f"{key}_wins",
        )

        losses = st.number_input(
            "Previous losses",
            min_value=0,
            value=int(record["losses"]),
            key=f"{key}_losses",
        )

    with c2:
        draws = st.number_input(
            "Previous draws",
            min_value=0,
            value=int(record["draws"]),
            key=f"{key}_draws",
        )

        height_default = safe_number(profile.get("height")) or 0.0

        height = st.number_input(
            "Height (cm)",
            min_value=0.0,
            value=float(height_default),
            key=f"{key}_height",
        )

    with c3:
        weight_default = safe_number(profile.get("weight")) or 0.0

        weight = st.number_input(
            "Weight (lbs)",
            min_value=0.0,
            value=float(weight_default),
            key=f"{key}_weight",
        )

        reach_default = safe_number(profile.get("reach")) or 0.0

        reach = st.number_input(
            "Reach (cm)",
            min_value=0.0,
            value=float(reach_default),
            key=f"{key}_reach",
        )

    stance_options = [
        "Unknown",
        "Orthodox",
        "Southpaw",
        "Switch",
        "Open Stance",
    ]

    current_stance = format_text(profile.get("stance"))

    if current_stance not in stance_options:
        current_stance = "Unknown"

    stance = st.selectbox(
        "Stance",
        stance_options,
        index=stance_options.index(current_stance),
        key=f"{key}_stance",
    )

    return {
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "fights": int(wins + losses + draws),
        "height": float(height) if height > 0 else None,
        "weight": float(weight) if weight > 0 else None,
        "reach": float(reach) if reach > 0 else None,
        "stance": stance,
    }


with st.expander("Edit Fighter A statistics", expanded=False):
    stats_a = fighter_prediction_inputs(
        fighter_a, profile_a, record_a, "a"
    )

with st.expander("Edit Fighter B statistics", expanded=False):
    stats_b = fighter_prediction_inputs(
        fighter_b, profile_b, record_b, "b"
    )


# =========================================================
# PREDICTION
# =========================================================

st.header("6. AI Fight Prediction")

st.warning(
    "Model probabilities are estimates, not guarantees. "
    "They depend on model limitations and the accuracy of input features."
)


def build_payload(a, b):
    return {
        "f_1_previous_wins": a["wins"],
        "f_1_previous_losses": a["losses"],
        "f_1_previous_draws": a["draws"],
        "f_1_previous_fights": a["fights"],

        "f_2_previous_wins": b["wins"],
        "f_2_previous_losses": b["losses"],
        "f_2_previous_draws": b["draws"],
        "f_2_previous_fights": b["fights"],

        "f_1_height": a["height"],
        "f_2_height": b["height"],

        "f_1_fighter_weight_lbs": a["weight"],
        "f_2_fighter_weight_lbs": b["weight"],

        "f_1_fighter_reach_cm": a["reach"],
        "f_2_fighter_reach_cm": b["reach"],

        "weight_class": selected_weight_class,
        "gender": gender,
        "num_rounds": num_rounds,
        "title_fight": title_fight,

        "f_1_fighter_stance": a["stance"],
        "f_2_fighter_stance": b["stance"],
    }


if st.button("🥊 Predict Winner", type="primary", use_container_width=True):

    if fighter_a == fighter_b:
        st.error("Choose two different fighters.")

    else:
        try:
            with st.spinner("Analyzing matchup..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=build_payload(stats_a, stats_b),
                    timeout=30,
                )

            if not response.ok:
                st.error(f"API error: {response.status_code}")
                st.code(response.text)

            else:
                result = response.json()

                probability_a = float(result["fighter_1_probability"])
                probability_b = float(result["fighter_2_probability"])

                if result["predicted_winner"] == "fighter_1":
                    winner_name = fighter_a
                elif result["predicted_winner"] == "fighter_2":
                    winner_name = fighter_b
                else:
                    st.error("Unexpected predicted_winner value.")
                    st.stop()

                winner_probability = max(probability_a, probability_b)

                st.markdown("---")
                st.subheader("🏆 Prediction Result")

                st.markdown(
                    f"""
                    <div class="winner-card">
                        <div class="winner-label">PREDICTED WINNER</div>
                        <div class="winner-name">{winner_name}</div>
                        <div class="winner-probability">
                            {winner_probability:.1%}
                        </div>
                        <p style="color:#D0D0D0;">
                            Model-estimated win probability
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                result_a, result_b = st.columns(2)

                with result_a:
                    st.markdown(f"### 🔴 {fighter_a}")
                    st.metric(
                        "Win probability",
                        f"{probability_a:.1%}",
                    )
                    st.progress(
                        min(max(probability_a, 0.0), 1.0)
                    )

                with result_b:
                    st.markdown(f"### 🔵 {fighter_b}")
                    st.metric(
                        "Win probability",
                        f"{probability_b:.1%}",
                    )
                    st.progress(
                        min(max(probability_b, 0.0), 1.0)
                    )

                st.caption(
                    "These probabilities are model estimates, "
                    "not guarantees of the actual fight outcome."
                )

        except requests.RequestException as exc:
            st.error(f"Could not connect to FastAPI: {exc}")

        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"Unexpected prediction response: {exc}")
            st.info(
                "Check the response schema returned by app/main.py."
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "OctagonIQ | UFC Fight Prediction | Educational ML Project"
)