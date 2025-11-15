from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from models.train_models import clean_frame

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "artifacts" / "model_registry.json"
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

SERVICE_COLUMNS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]
TIME_COLUMNS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
CATEGORICAL_COLUMNS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMERIC_COLUMNS = ["Age", "Flight Distance", *SERVICE_COLUMNS, *TIME_COLUMNS]
QUESTION_SECTIONS = ["Trip basics", "Onboard experience", "Ops & service"]

st.set_page_config(
    page_title="Passenger Experience Studio",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_registry() -> Dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            "Model registry missing. Run src/models/train_models.py first."
        )
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_models(registry: Dict) -> Dict[str, any]:
    loaded = {}
    for entry in registry.get("models", []):
        artifact = Path(entry["artifact_path"])
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        try:
            loaded[entry["name"]] = joblib.load(artifact)
        except FileNotFoundError:
            st.warning(f"Missing artifact for {entry['name']} at {artifact}")
    return loaded


@st.cache_data(show_spinner=False)
def load_reference_summary() -> Dict:
    df = pd.read_csv(TRAIN_PATH)
    df = clean_frame(df)
    defaults: Dict[str, any] = {}
    numeric_stats: Dict[str, Dict[str, float]] = {}
    categorical_options: Dict[str, List[str]] = {}
    categorical_freq: Dict[Tuple[str, str], float] = {}

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        numeric_stats[col] = {
            "median": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max()),
        }
        defaults[col] = numeric_stats[col]["median"]

    for col in CATEGORICAL_COLUMNS:
        series = df[col].dropna()
        defaults[col] = series.mode().iat[0]
        options = sorted(series.unique().tolist())
        categorical_options[col] = options
        freq = (series.value_counts(normalize=True)).to_dict()
        categorical_freq.update({(col, k): float(v) for k, v in freq.items()})

    defaults.setdefault("Age", 35)
    defaults.setdefault("Flight Distance", 1000)

    return {
        "defaults": defaults,
        "numeric_stats": numeric_stats,
        "categorical_options": categorical_options,
        "categorical_freq": categorical_freq,
    }


def ensure_session_state(summary: Dict) -> None:
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 0
    if "responses" not in st.session_state:
        st.session_state.responses = summary["defaults"].copy()


def wizard_navigation(total_steps: int) -> None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state.wizard_step > 0:
            if st.button("◀ Back", use_container_width=True):
                st.session_state.wizard_step -= 1
    with col3:
        if st.session_state.wizard_step < total_steps - 1:
            if st.button("Next ▶", use_container_width=True):
                st.session_state.wizard_step += 1

    st.progress((st.session_state.wizard_step + 1) / total_steps)


def render_trip_basics(summary: Dict) -> None:
    responses = st.session_state.responses
    options = summary["categorical_options"]
    col1, col2 = st.columns(2)
    with col1:
        responses["Gender"] = st.radio(
            "Passenger gender",
            options["Gender"],
            horizontal=True,
            index=options["Gender"].index(responses.get("Gender", options["Gender"][0])),
        )
        responses["Customer Type"] = st.selectbox(
            "Customer tenure",
            options["Customer Type"],
            index=options["Customer Type"].index(
                responses.get("Customer Type", options["Customer Type"][0])
            ),
        )
        responses["Type of Travel"] = st.selectbox(
            "Travel purpose",
            options["Type of Travel"],
            index=options["Type of Travel"].index(
                responses.get("Type of Travel", options["Type of Travel"][0])
            ),
        )
    with col2:
        stats = summary["numeric_stats"]["Age"]
        responses["Age"] = st.slider(
            "Age",
            int(stats["min"]),
            int(stats["max"]),
            int(responses.get("Age", stats["median"])),
        )
        responses["Class"] = st.segmented_control(
            "Cabin class",
            options["Class"],
            selection_mode="single",
            default=responses.get("Class", options["Class"][0]),
        )
        stats = summary["numeric_stats"]["Flight Distance"]
        responses["Flight Distance"] = st.slider(
            "Flight distance (miles)",
            int(stats["min"]),
            int(stats["max"]),
            int(responses.get("Flight Distance", stats["median"])),
            help="Total distance flown for this itinerary.",
        )


def render_experience_section(summary: Dict) -> None:
    responses = st.session_state.responses
    st.caption("Rate each touchpoint (0 = not applicable, 5 = delighted)")
    cols = st.columns(3)
    for idx, feature in enumerate(SERVICE_COLUMNS):
        col = cols[idx % 3]
        with col:
            stats = summary["numeric_stats"].get(feature, {"median": 3, "min": 0, "max": 5})
            responses[feature] = st.slider(
                feature,
                int(stats["min"]),
                int(stats["max"]),
                int(responses.get(feature, stats["median"])),
            )


def render_ops_section(summary: Dict) -> None:
    responses = st.session_state.responses
    col1, col2 = st.columns(2)
    for col, feature in zip((col1, col2), TIME_COLUMNS):
        stats = summary["numeric_stats"].get(feature, {"median": 5, "min": 0, "max": 120})
        with col:
            responses[feature] = st.slider(
                feature,
                int(stats["min"]),
                int(max(stats["max"], stats["median"] + 1)),
                int(responses.get(feature, stats["median"])),
                help="Minutes of delay for this flight stage.",
            )
    st.divider()
    st.subheader("Ground & baggage service")
    cols = st.columns(2)
    for idx, feature in enumerate(["Baggage handling", "Checkin service"]):
        stats = summary["numeric_stats"].get(feature, {"median": 3, "min": 0, "max": 5})
        with cols[idx]:
            responses[feature] = st.slider(
                feature,
                int(stats["min"]),
                int(stats["max"]),
                int(responses.get(feature, stats["median"])),
            )


def format_probability(prob: float) -> Tuple[str, str]:
    label = "satisfied" if prob >= 0.5 else "neutral / dissatisfied"
    emoji = "✅" if prob >= 0.5 else "⚠️"
    return label, emoji


def predict_probability(model, model_name: str, features: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0, 1])
    if hasattr(model, "decision_function"):
        raw = model.decision_function(features)
        raw = np.asarray(raw)
        if raw.ndim > 1:
            raw = raw[:, -1]
        raw_min, raw_max = raw.min(), raw.max()
        span = raw_max - raw_min
        if span == 0:
            span = 1.0
        return float((raw - raw_min) / span)
    raw = model.predict(features)
    if model_name in {"linear_regression", "polynomial_regression"}:
        raw = np.clip(raw, 0, 1)
    return float(np.asarray(raw).ravel()[0])


def canonicalize_feature(name: str) -> str:
    clean = name.replace("cat__", "").replace("num__", "")
    return clean


def compute_personal_impacts(
    registry: Dict,
    responses: Dict,
    summary: Dict,
) -> List[Dict[str, float]]:
    weight_map: Dict[str, float] = {}
    log_model = next(
        (model for model in registry.get("models", []) if model["name"] == "logistic_regression"),
        None,
    )
    if log_model:
        for item in log_model.get("feature_importances", []):
            canonical = canonicalize_feature(item["feature"])
            weight_map[canonical] = weight_map.get(canonical, 0.0) + item["weight"]

    impacts: List[Dict[str, float]] = []
    for feature, value in responses.items():
        if feature in CATEGORICAL_COLUMNS:
            key = (feature, value)
            baseline = summary["categorical_freq"].get(key, 0.5)
            delta = 1.0 - baseline
        else:
            stats = summary["numeric_stats"].get(feature)
            if not stats:
                continue
            denom = max(stats["max"] - stats["min"], 1)
            delta = (value - stats["median"]) / denom
        weight = weight_map.get(feature, 0.0)
        impacts.append(
            {
                "feature": feature,
                "impact": float(delta * weight),
                "weight": weight,
                "value": value,
            }
        )
    impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return impacts[:6]


def _impact_severity_label(impact: float) -> str:
    value = abs(impact)
    if value >= 0.2:
        return "critical"
    if value >= 0.1:
        return "major"
    return "noticeable"


def _format_feature_action(feature: str, responses: Dict, impact: float) -> str:
    severity = _impact_severity_label(impact)
    value = responses.get(feature)
    if feature in SERVICE_COLUMNS:
        return (
            f"{severity.title()} drag on **{feature}** ({value}/5). "
            "Invest in coaching, staffing, or product tweaks to lift this touchpoint."
        )
    if feature in TIME_COLUMNS and value is not None:
        return (
            f"{severity.title()} delay risk: **{feature}** sits at {int(value)} min. "
            "Align ops control and gate teams to collapse turnaround times."
        )
    if feature in {"Customer Type", "Type of Travel", "Class"}:
        return (
            f"{severity.title()} friction for **{feature} = {value}**. "
            "Adjust targeting or perks for this segment to avoid churn."
        )
    if feature == "Gender":
        return (
            f"{severity.title()} sentiment gap detected across gender expression ({value}). "
            "Ensure comms and service scripts stay inclusive."
        )
    return (
        f"{severity.title()} headwind on **{feature}** (response: {value}). "
        "Deep dive qualitative feedback to diagnose root causes."
    )


def build_suggestions(
    responses: Dict,
    impacts: List[Dict[str, float]],
    probability: float,
    max_items: int = 4,
) -> List[str]:
    suggestions: List[str] = []
    negative_impacts = [imp for imp in impacts if imp["impact"] < 0]
    negative_impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)

    for impact_record in negative_impacts[:max_items]:
        feature = impact_record["feature"]
        if feature not in responses:
            continue
        suggestions.append(
            _format_feature_action(feature, responses, impact_record["impact"])
        )

    if probability < 0.5:
        suggestions.insert(
            0,
            "Activate service-recovery playbook (personal outreach, vouchers, and proactive status updates).",
        )
    elif probability >= 0.8:
        suggestions.append(
            "Leverage promoter energy—invite the traveler to referral or loyalty-upgrade campaigns."
        )

    if not suggestions:
        suggestions.append(
            "Maintain current service blueprint—passenger signaled no urgent friction points."
        )
    return suggestions


def model_metrics_frame(registry: Dict, dataset: str = "test") -> pd.DataFrame:
    records = []
    for model in registry.get("models", []):
        metrics = model["metrics"].get(dataset, {})
        records.append(
            {
                "model": model["name"],
                **metrics,
            }
        )
    return pd.DataFrame(records)


def collect_model_scores(
    models: Dict[str, Any],
    features: pd.DataFrame,
    base_rate: float,
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        prob = predict_probability(model, name, features)
        label, emoji = format_probability(prob)
        rows.append(
            {
                "model": name,
                "probability": prob,
                "probability_label": f"{prob:.1%}",
                "lift": prob - base_rate,
                "label": label,
                "emoji": emoji,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("probability", ascending=False).reset_index(drop=True)


def next_best_action(probability: float) -> Tuple[str, str]:
    if probability >= 0.8:
        return (
            "success",
            "Celebrate the advocate—surprise-and-delight perks or loyalty nudges will reinforce satisfaction.",
        )
    if probability >= 0.5:
        return (
            "info",
            "Solid but fragile sentiment—reinforce digital touchpoints and follow up with a post-trip survey.",
        )
    return (
        "warning",
        "High churn risk—trigger service recovery workflow and contact the passenger proactively.",
    )


def render_metrics_tab(registry: Dict) -> None:
    st.subheader("Model diagnostics")
    dataset_choice = st.radio(
        "Dataset",
        options=["train", "test"],
        horizontal=True,
        index=1,
    )
    df = model_metrics_frame(registry, dataset_choice)
    if df.empty:
        st.info("Train models first to inspect diagnostics.")
        return
    selected_model = st.selectbox("Inspect model", df["model"].tolist())
    row = df[df["model"] == selected_model].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{row['accuracy']:.3f}")
    c2.metric("Precision", f"{row['precision']:.3f}")
    c3.metric("Recall", f"{row['recall']:.3f}")
    c4.metric("F1", f"{row['f1']:.3f}")

    chart_df = df.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"])
    fig = px.bar(
        chart_df,
        x="value",
        y="model",
        color="variable",
        orientation="h",
        barmode="group",
        title=f"{dataset_choice.title()} metrics",
        range_x=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)

    comparison = st.multiselect(
        "Compare models",
        df["model"].tolist(),
        default=df["model"].tolist()[:3],
    )
    if comparison:
        st.dataframe(df[df["model"].isin(comparison)].set_index("model"))

    selected_registry = next(
        model for model in registry["models"] if model["name"] == selected_model
    )
    cm = np.array(selected_registry["confusion_matrices"][dataset_choice])
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["Neutral", "Satisfied"],
        y=["Neutral", "Satisfied"],
        color_continuous_scale="Blues",
        title=f"{selected_model} confusion matrix ({dataset_choice})",
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    with st.expander("Cross-validation (mean ± std)"):
        cv = selected_registry.get("cv_metrics", {})
        mean = cv.get("mean", {})
        std = cv.get("std", {})
        table = pd.DataFrame(
            {
                "metric": list(mean.keys()),
                "mean": [mean[m] for m in mean],
                "std": [std.get(m, math.nan) for m in mean],
            }
        )
        st.dataframe(table.set_index("metric"))


@st.cache_data(show_spinner=False)
def load_registry_models() -> Tuple[Dict, Dict[str, any]]:
    registry = load_registry()
    models = load_models(registry)
    return registry, models


registry, models = load_registry_models()
summary = load_reference_summary()
ensure_session_state(summary)

base_rate = registry.get("base_rate", 0.5)
model_names = list(models.keys())
if not model_names:
    st.error("No trained models loaded. Train them first.")
    st.stop()

with st.sidebar:
    st.header("Score control")
    selected_model_name = st.selectbox(
        "Model for scoring",
        model_names,
        help="Switch between pipelines to see how predictions change.",
    )
    st.metric("Training base rate", f"{base_rate:.2%}")
    best_model = registry.get("best_model", model_names[0])
    st.info(f"Registry champion: **{best_model}**")

st.title("Passenger Experience Studio")
st.write(
    "Guided intake captures each touchpoint, then multiple models forecast satisfaction with full transparency."
)

user_tab, metrics_tab, insights_tab = st.tabs(
    ["Passenger wizard", "Model lab", "Global insights"]
)

with user_tab:
    st.subheader("Passenger journey wizard")
    wizard_navigation(len(QUESTION_SECTIONS))
    step = st.session_state.wizard_step
    if step == 0:
        render_trip_basics(summary)
    elif step == 1:
        render_experience_section(summary)
    else:
        render_ops_section(summary)

    captured = st.session_state.responses
    with st.expander("Current responses", expanded=False):
        st.json(captured)

    if step == len(QUESTION_SECTIONS) - 1:
        if st.button("Generate satisfaction score", type="primary", use_container_width=True):
            features = pd.DataFrame([captured])
            with st.spinner("Scoring across selected pipeline..."):
                score_df = collect_model_scores(models, features, base_rate)

            if score_df.empty:
                st.error("Unable to score—no models were loaded.")
                st.stop()

            current_row = score_df[score_df["model"] == selected_model_name].iloc[0]
            probability = float(current_row["probability"])
            label = current_row["label"]
            emoji = current_row["emoji"]

            st.success(
                f"{emoji} {selected_model_name} projects **{probability:.1%}** chance of a {label} traveler."
            )
            st.progress(probability)
            st.metric(
                "Lift vs. base rate",
                f"{probability - base_rate:+.1%}",
                help="Positive lift means the traveler is more likely to be satisfied than the historic average.",
            )

            severity, action_copy = next_best_action(probability)
            getattr(st, severity)(f"Next best action: {action_copy}")

            st.write("#### Model agreement")
            display_df = score_df.copy()
            display_df["probability"] = display_df["probability"].map(lambda x: f"{x:.1%}")
            display_df["lift"] = display_df["lift"].map(lambda x: f"{x:+.1%}")
            st.dataframe(
                display_df.set_index("model")[
                    ["probability", "lift", "label"]
                ],
                use_container_width=True,
            )

            chart_df = score_df.assign(model_rank=lambda df: df.index)
            fig_scores = px.bar(
                chart_df,
                x="probability",
                y="model",
                orientation="h",
                text="probability_label",
                range_x=[0, 1],
                title="How each model scored this traveler",
                color="model",
            )
            fig_scores.update_layout(showlegend=False)
            st.plotly_chart(fig_scores, use_container_width=True)

            impacts = compute_personal_impacts(registry, captured, summary)
            if impacts:
                impact_df = pd.DataFrame(impacts)
                impact_fig = px.bar(
                    impact_df,
                    x="impact",
                    y="feature",
                    color="impact",
                    orientation="h",
                    title="Top personalized drivers",
                    color_continuous_scale="Tealrose",
                )
                st.plotly_chart(impact_fig, use_container_width=True)

            with st.popover("Why these drivers?"):
                st.write(
                    "We project influence by blending logistic-regression coefficients with how far each response deviates from the survey median."
                )

            suggestions = build_suggestions(captured, impacts or [], probability)
            st.write("### Ops cockpit recommendations")
            for tip in suggestions:
                st.write(f"- {tip}")

with metrics_tab:
    render_metrics_tab(registry)

with insights_tab:
    st.subheader("Global signals")
    st.metric("Best model", registry.get("best_model", "n/a"))
    top_features = registry.get("global_feature_rank", [])
    if top_features:
        df = pd.DataFrame(top_features)
        df.rename(columns={"feature": "Feature", "importance": "Normalized impact"}, inplace=True)
        st.dataframe(df.set_index("Feature"))
        fig = px.bar(
            df,
            x="Normalized impact",
            y="Feature",
            title="Across-model importance",
            orientation="h",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train models to see feature rankings.")

    with st.expander("Data dictionary"):
        st.markdown(
            textwrap.dedent(
                """
                - **Service sliders**: all 0-5 where 5 = delighted.
                - **Class**: Business, Eco, Eco Plus as per airline ticketing.
                - **Satisfaction label**: 1 = satisfied, 0 = neutral/dissatisfied.
                """
            )
        )

    st.caption(
        "App built with Streamlit 1.51 tabs, progress indicators, popovers, and responsive layout for seamless mobile & desktop use."
    )
