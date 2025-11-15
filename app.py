from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.train_models import clean_frame

REGISTRY_PATH = PROJECT_ROOT / "artifacts" / "model_registry.json"
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
LOCALES_DIR = PROJECT_ROOT / "locales"
DEFAULT_LANGUAGE = "en"

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
QUESTION_SECTIONS = ["trip_basics", "experience", "operations"]
PAGE_CODES = ["wizard", "analytics", "documentation"]
DOC_LOCALE_SUFFIX = "_doc"


def load_translation_files() -> Dict[str, Dict[str, Any]]:
    translations: Dict[str, Dict[str, Any]] = {}
    if not LOCALES_DIR.exists():
        raise FileNotFoundError(f"Missing locales directory at {LOCALES_DIR}")
    for path in LOCALES_DIR.glob("*.json"):
        with path.open("r", encoding="utf-8") as handle:
            translations[path.stem] = json.load(handle)
    return translations


TRANSLATIONS = load_translation_files()
if DEFAULT_LANGUAGE not in TRANSLATIONS:
    raise FileNotFoundError("Default locale 'en.json' is required in locales/")

DEFAULT_LOCALE = TRANSLATIONS[DEFAULT_LANGUAGE]

st.set_page_config(
    page_title=DEFAULT_LOCALE["app"]["page_title"],
    page_icon=DEFAULT_LOCALE["app"].get("page_icon", "✈️"),
    layout="wide",
    initial_sidebar_state="expanded",
)


def translate(lang: str, key: str, default: Any | None = None, **kwargs) -> Any:
    def lookup(code: str) -> Any:
        data = TRANSLATIONS.get(code)
        if not data:
            return None
        node: Any = data
        for part in key.split("."):
            if isinstance(node, dict):
                node = node.get(part)
            else:
                return None
        return node

    value = lookup(lang)
    if value is None:
        value = lookup(DEFAULT_LANGUAGE)
    if value is None:
        value = default if default is not None else key
    if isinstance(value, str) and kwargs:
        try:
            return value.format(**kwargs)
        except KeyError:
            return value
    return value if value is not None else (default if default is not None else key)


def language_label(code: str) -> str:
    meta = TRANSLATIONS.get(code, {}).get("meta", {})
    return meta.get("language_name", code.upper())


def feature_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def feature_label(feature: str, lang: str) -> str:
    return translate(lang, f"features.{feature_slug(feature)}.label", default=feature)


def feature_help(feature: str, lang: str) -> str:
    return translate(lang, f"features.{feature_slug(feature)}.help", default="")


INPUT_KEY_MAP = {
    "Gender": "gender",
    "Customer Type": "customer_type",
    "Type of Travel": "type_of_travel",
    "Age": "age",
    "Class": "class",
    "Flight Distance": "flight_distance",
}


def input_label(field: str, lang: str) -> str:
    key = INPUT_KEY_MAP.get(field, field.lower())
    return translate(lang, f"inputs.{key}.label", default=field)


def input_help(field: str, lang: str) -> str:
    key = INPUT_KEY_MAP.get(field, field.lower())
    return translate(lang, f"inputs.{key}.help", default="")


def section_label(section_key: str, lang: str) -> str:
    return translate(lang, f"wizard.sections.{section_key}.label", default=section_key.title())


def section_description(section_key: str, lang: str) -> str:
    return translate(lang, f"wizard.sections.{section_key}.description", default="")


def page_label(code: str, lang: str) -> str:
    return translate(lang, f"sidebar.page_labels.{code}", default=code.title())


def dataset_label(code: str, lang: str) -> str:
    return translate(lang, f"datasets.{code}", default=code.title())


def metric_label(metric: str, lang: str) -> str:
    return translate(lang, f"metrics.{metric}", default=metric.upper())


def confusion_axis(axis: str, lang: str) -> str:
    return translate(lang, f"confusion_axes.{axis}", default=axis.title())


def probability_label(prob: float, lang: str) -> Tuple[str, str]:
    key = "satisfied" if prob >= 0.5 else "neutral"
    label = translate(lang, f"probability_labels.{key}", default=key)
    emoji = "✅" if prob >= 0.5 else "⚠️"
    return label, emoji


def _resolve_artifact_path(raw_path: str) -> Path:
    clean = raw_path.replace("\\", "/")
    artifact = Path(clean)
    if not artifact.is_absolute():
        artifact = PROJECT_ROOT / artifact
    return artifact


@st.cache_data(show_spinner=False)
def load_registry() -> Dict[str, Any] | None:
    if not REGISTRY_PATH.exists():
        return None
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_models(registry: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Tuple[str, Path]]]:
    loaded: Dict[str, Any] = {}
    missing: List[Tuple[str, Path]] = []
    for entry in registry.get("models", []):
        artifact = _resolve_artifact_path(entry["artifact_path"])
        try:
            loaded[entry["name"]] = joblib.load(artifact)
        except FileNotFoundError:
            missing.append((entry["name"], artifact))
    return loaded, missing


@st.cache_data(show_spinner=False)
def load_reference_summary() -> Dict[str, Any]:
    df = clean_frame(pd.read_csv(TRAIN_PATH))
    defaults: Dict[str, Any] = {}
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
        if col not in df.columns:
            continue
        series = df[col].dropna()
        defaults[col] = series.mode().iat[0]
        categorical_options[col] = sorted(series.unique().tolist())
        freq = series.value_counts(normalize=True).to_dict()
        categorical_freq.update({(col, k): float(v) for k, v in freq.items()})

    defaults.setdefault("Age", 35)
    defaults.setdefault("Flight Distance", 1000)

    return {
        "defaults": defaults,
        "numeric_stats": numeric_stats,
        "categorical_options": categorical_options,
        "categorical_freq": categorical_freq,
    }


def ensure_session_state(summary: Dict[str, Any]) -> None:
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 0
    if "responses" not in st.session_state:
        st.session_state.responses = summary["defaults"].copy()


def wizard_navigation(total_steps: int, lang: str) -> None:
    col1, _, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state.wizard_step > 0:
            if st.button(translate(lang, "wizard.back_button"), use_container_width=True):
                st.session_state.wizard_step -= 1
    with col3:
        if st.session_state.wizard_step < total_steps - 1:
            if st.button(translate(lang, "wizard.next_button"), use_container_width=True):
                st.session_state.wizard_step += 1
    st.progress((st.session_state.wizard_step + 1) / total_steps)
    st.caption(translate(lang, "wizard.progress_caption"))


def render_trip_basics(summary: Dict[str, Any], lang: str) -> None:
    responses = st.session_state.responses
    options = summary["categorical_options"]
    col1, col2 = st.columns(2)
    with col1:
        responses["Gender"] = st.radio(
            input_label("Gender", lang),
            options["Gender"],
            horizontal=True,
            index=options["Gender"].index(responses.get("Gender", options["Gender"][0])),
            help=input_help("Gender", lang),
        )
        responses["Customer Type"] = st.selectbox(
            input_label("Customer Type", lang),
            options["Customer Type"],
            index=options["Customer Type"].index(
                responses.get("Customer Type", options["Customer Type"][0])
            ),
            help=input_help("Customer Type", lang),
        )
        responses["Type of Travel"] = st.selectbox(
            input_label("Type of Travel", lang),
            options["Type of Travel"],
            index=options["Type of Travel"].index(
                responses.get("Type of Travel", options["Type of Travel"][0])
            ),
            help=input_help("Type of Travel", lang),
        )
    with col2:
        stats = summary["numeric_stats"].get("Age", {"min": 0, "max": 80, "median": 35})
        responses["Age"] = st.slider(
            input_label("Age", lang),
            int(stats["min"]),
            int(stats["max"]),
            int(responses.get("Age", stats["median"])),
            help=input_help("Age", lang),
        )
        responses["Class"] = st.segmented_control(
            input_label("Class", lang),
            options["Class"],
            selection_mode="single",
            default=responses.get("Class", options["Class"][0]),
        )
        stats = summary["numeric_stats"].get("Flight Distance", {"min": 0, "max": 4000, "median": 1000})
        responses["Flight Distance"] = st.slider(
            input_label("Flight Distance", lang),
            int(stats["min"]),
            int(stats["max"]),
            int(responses.get("Flight Distance", stats["median"])),
            help=input_help("Flight Distance", lang),
        )


def render_experience_section(summary: Dict[str, Any], lang: str) -> None:
    responses = st.session_state.responses
    cols = st.columns(3)
    for idx, feature in enumerate(SERVICE_COLUMNS):
        stats = summary["numeric_stats"].get(feature, {"median": 3, "min": 0, "max": 5})
        col = cols[idx % 3]
        with col:
            responses[feature] = st.slider(
                feature_label(feature, lang),
                int(stats["min"]),
                int(stats["max"]),
                int(responses.get(feature, stats["median"])),
                help=feature_help(feature, lang),
            )


def render_ops_section(summary: Dict[str, Any], lang: str) -> None:
    responses = st.session_state.responses
    col1, col2 = st.columns(2)
    for col, feature in zip((col1, col2), TIME_COLUMNS):
        stats = summary["numeric_stats"].get(feature, {"median": 5, "min": 0, "max": 120})
        with col:
            responses[feature] = st.slider(
                feature_label(feature, lang),
                int(stats["min"]),
                int(max(stats["max"], stats["median"] + 1)),
                int(responses.get(feature, stats["median"])),
                help=feature_help(feature, lang),
            )
    st.divider()
    st.subheader(translate(lang, "wizard.ground_section_title"))
    st.caption(translate(lang, "inputs.ground_section.help"))
    cols = st.columns(2)
    for idx, feature in enumerate(["Baggage handling", "Checkin service"]):
        stats = summary["numeric_stats"].get(feature, {"median": 3, "min": 0, "max": 5})
        with cols[idx]:
            responses[feature] = st.slider(
                feature_label(feature, lang),
                int(stats["min"]),
                int(stats["max"]),
                int(responses.get(feature, stats["median"])),
                help=feature_help(feature, lang),
            )


def predict_probability(model: Any, model_name: str, features: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0, 1])
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(features))
        if raw.ndim > 1:
            raw = raw[:, -1]
        raw_min, raw_max = float(raw.min()), float(raw.max())
        span = raw_max - raw_min or 1.0
        return float((raw - raw_min) / span)
    raw = model.predict(features)
    if model_name in {"linear_regression", "polynomial_regression"}:
        raw = np.clip(raw, 0, 1)
    return float(np.asarray(raw).ravel()[0])


def canonicalize_feature(name: str) -> str:
    return name.replace("cat__", "").replace("num__", "")


def compute_personal_impacts(
    registry: Dict[str, Any],
    responses: Dict[str, Any],
    summary: Dict[str, Any],
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
            baseline = summary["categorical_freq"].get((feature, value), 0.5)
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


def _impact_severity_key(impact: float) -> str:
    value = abs(impact)
    if value >= 0.2:
        return "critical"
    if value >= 0.1:
        return "major"
    return "noticeable"


def _suggestion_template_key(feature: str) -> str:
    if feature in SERVICE_COLUMNS:
        return "service"
    if feature in TIME_COLUMNS:
        return "delay"
    if feature in {"Customer Type", "Type of Travel", "Class"}:
        return "segment"
    if feature == "Gender":
        return "gender"
    return "generic"


def build_suggestions(
    responses: Dict[str, Any],
    impacts: List[Dict[str, float]],
    probability: float,
    lang: str,
    max_items: int = 4,
) -> List[str]:
    suggestions: List[str] = []
    negative_impacts = [imp for imp in impacts if imp["impact"] < 0]
    negative_impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)

    for impact_record in negative_impacts[:max_items]:
        feature = impact_record["feature"]
        if feature not in responses:
            continue
        severity_key = _impact_severity_key(impact_record["impact"])
        severity_label = translate(lang, f"suggestions.severity.{severity_key}", default=severity_key.title())
        template_key = _suggestion_template_key(feature)
        template = translate(lang, f"suggestions.templates.{template_key}")
        if not template:
            continue
        value_display = responses.get(feature)
        suggestions.append(
            template.format(
                severity=severity_label,
                feature=feature_label(feature, lang),
                value=value_display,
            )
        )

    if probability < 0.5:
        suggestions.insert(0, translate(lang, "suggestions.recovery_action"))
    elif probability >= 0.8:
        suggestions.append(translate(lang, "suggestions.promoter_action"))

    if not suggestions:
        suggestions.append(translate(lang, "suggestions.no_action"))
    return suggestions


def model_metrics_frame(registry: Dict[str, Any], dataset: str = "test") -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for model in registry.get("models", []):
        metrics = model.get("metrics", {}).get(dataset, {})
        records.append({"model": model["name"], **metrics})
    return pd.DataFrame(records)


def collect_model_scores(
    models: Dict[str, Any],
    features: pd.DataFrame,
    base_rate: float,
    lang: str,
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        prob = predict_probability(model, name, features)
        label, emoji = probability_label(prob, lang)
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


def next_best_action(probability: float, lang: str) -> Tuple[str, str]:
    if probability >= 0.8:
        key = "success"
    elif probability >= 0.5:
        key = "info"
    else:
        key = "warning"
    return key, translate(lang, f"next_best_action.{key}")


def render_model_diagnostics(registry: Dict[str, Any], lang: str) -> None:
    st.subheader(translate(lang, "analytics.model_section_title"))
    dataset_choice = st.radio(
        translate(lang, "analytics.dataset_label"),
        options=["train", "test"],
        horizontal=True,
        index=1,
        format_func=lambda code: dataset_label(code, lang),
    )
    df = model_metrics_frame(registry, dataset_choice)
    if df.empty:
        st.info(translate(lang, "analytics.global_table_empty"))
        return
    selected_model = st.selectbox(
        translate(lang, "analytics.model_dropdown"),
        df["model"].tolist(),
    )
    row = df[df["model"] == selected_model].iloc[0]
    metrics_to_show = ["accuracy", "precision", "recall", "f1", "roc_auc", "r2"]
    cols = st.columns(len(metrics_to_show))
    for col, metric in zip(cols, metrics_to_show):
        value = row.get(metric)
        if value is None or isinstance(value, str):
            display_value = "–"
        else:
            display_value = f"{value:.3f}"
        col.metric(metric_label(metric, lang), display_value)

    chart_metrics = [m for m in metrics_to_show if m in df.columns]
    chart_df = df.melt(id_vars="model", value_vars=chart_metrics)
    if not chart_df.empty:
        min_val = chart_df["value"].min()
        max_val = chart_df["value"].max()
        x_min = min(0.0, float(min_val)) if pd.notna(min_val) else 0.0
        x_max = max(1.0, float(max_val)) if pd.notna(max_val) else 1.0
        fig = px.bar(
            chart_df,
            x="value",
            y="model",
            color="variable",
            orientation="h",
            barmode="group",
            title=translate(lang, "analytics.chart_title", dataset=dataset_label(dataset_choice, lang)),
            range_x=[x_min, x_max],
        )
        st.plotly_chart(fig, use_container_width=True)
    st.caption(translate(lang, "analytics.metrics_help"))

    comparison = st.multiselect(
        translate(lang, "analytics.compare_label"),
        df["model"].tolist(),
        default=df["model"].tolist()[:3],
        help=translate(lang, "analytics.comparison_help"),
    )
    if comparison:
        st.dataframe(df[df["model"].isin(comparison)].set_index("model"))

    selected_registry = next(model for model in registry["models"] if model["name"] == selected_model)
    cm = np.array(selected_registry["confusion_matrices"][dataset_choice])
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x=confusion_axis("predicted", lang), y=confusion_axis("actual", lang)),
        x=[confusion_axis("neutral", lang), confusion_axis("satisfied", lang)],
        y=[confusion_axis("neutral", lang), confusion_axis("satisfied", lang)],
        color_continuous_scale="Blues",
        title=translate(
            lang,
            "analytics.confusion_title",
            model=selected_model,
            dataset=dataset_label(dataset_choice, lang),
        ),
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    with st.expander(translate(lang, "analytics.cv_expander")):
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


def render_global_signals(registry: Dict[str, Any], lang: str) -> None:
    st.subheader(translate(lang, "analytics.global_section_title"))
    st.metric(translate(lang, "analytics.best_model_label"), registry.get("best_model", "n/a"))
    top_features = registry.get("global_feature_rank", [])
    if top_features:
        df = pd.DataFrame(top_features)
        df.rename(columns={"feature": "Feature", "importance": "Normalized impact"}, inplace=True)
        st.dataframe(df.set_index("Feature"))
        fig = px.bar(
            df,
            x="Normalized impact",
            y="Feature",
            title=translate(lang, "analytics.global_section_title"),
            orientation="h",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(translate(lang, "analytics.global_help"))
    else:
        st.info(translate(lang, "analytics.global_table_empty"))

    with st.expander(translate(lang, "analytics.data_dictionary_title")):
        lines = translate(lang, "analytics.data_dictionary_lines", default=[])
        if isinstance(lines, list):
            for line in lines:
                st.markdown(f"- {line}")
        else:
            st.markdown(lines)


def load_doc_locales() -> Dict[str, Dict[str, Any]]:
    docs: Dict[str, Dict[str, Any]] = {}
    suffix_len = len(DOC_LOCALE_SUFFIX)
    for path in LOCALES_DIR.glob(f"*{DOC_LOCALE_SUFFIX}.json"):
        language_code = path.stem[:-suffix_len]
        with path.open("r", encoding="utf-8") as handle:
            docs[language_code] = json.load(handle)
    return docs


DOC_LOCALES = load_doc_locales()
if DEFAULT_LANGUAGE not in DOC_LOCALES:
    raise FileNotFoundError(
        f"Documentation locale '{DEFAULT_LANGUAGE}{DOC_LOCALE_SUFFIX}.json' is required in locales/"
    )


def get_doc_locale(lang: str) -> Dict[str, Any]:
    return DOC_LOCALES.get(lang) or DOC_LOCALES[DEFAULT_LANGUAGE]


def format_doc_value(value: Any, context: Dict[str, Any]) -> Any:
    if value is None:
        return value
    if isinstance(value, str):
        try:
            return value.format(**context)
        except KeyError:
            return value
    if isinstance(value, list):
        return [format_doc_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: format_doc_value(val, context) for key, val in value.items()}
    return value


def render_doc_title(level: str, title: str) -> None:
    if not title:
        return
    if level == "title":
        st.title(title)
    elif level == "header":
        st.header(title)
    elif level == "subheader":
        st.subheader(title)
    elif level == "caption":
        st.caption(title)
    else:
        st.markdown(f"**{title}**")


def render_doc_section(section: Dict[str, Any], context: Dict[str, Any]) -> None:
    section_type = section.get("type", "markdown")
    level = section.get("level", "subheader")
    title = format_doc_value(section.get("title", ""), context)

    if section_type not in {"callout", "divider", "expander"}:
        render_doc_title(level, title)

    if section_type == "markdown":
        body = format_doc_value(section.get("body", []), context)
        if isinstance(body, str):
            st.markdown(body)
        else:
            for line in body:
                st.markdown(line)
        caption = format_doc_value(section.get("caption"), context)
        if caption:
            st.caption(caption)
    elif section_type == "list":
        items = format_doc_value(section.get("items", []), context) or []
        for item in items:
            st.markdown(f"- {item}")
    elif section_type == "table":
        raw_columns = section.get("columns", [])
        columns: List[Dict[str, str]] = []
        for col in raw_columns:
            if isinstance(col, str):
                columns.append({"key": col, "label": format_doc_value(col, context)})
            else:
                label = format_doc_value(col.get("label", col.get("key", "")), context)
                columns.append({"key": col.get("key", label), "label": label})
        rows: List[Dict[str, Any]] = []
        for row in section.get("rows", []):
            formatted_row: Dict[str, Any] = {}
            for col in columns:
                formatted_row[col["label"]] = format_doc_value(row.get(col["key"]), context)
            rows.append(formatted_row)
        df = pd.DataFrame(rows)
        index_key = section.get("index_column")
        if index_key:
            label_lookup = next((col["label"] for col in columns if col["key"] == index_key), None)
            if label_lookup and label_lookup in df.columns:
                df = df.set_index(label_lookup)
        st.dataframe(df, use_container_width=True)
        caption = format_doc_value(section.get("caption"), context)
        if caption:
            st.caption(caption)
        footer = format_doc_value(section.get("footer"), context)
        if footer:
            st.caption(footer)
        expander = section.get("expander")
        if expander:
            with st.expander(format_doc_value(expander.get("title", "Details"), context)):
                body = format_doc_value(expander.get("body", []), context)
                if isinstance(body, str):
                    st.markdown(body)
                else:
                    for line in body:
                        st.markdown(line)
    elif section_type == "expander":
        with st.expander(title or "Details"):
            body = format_doc_value(section.get("body", []), context)
            if isinstance(body, str):
                st.markdown(body)
            else:
                for line in body:
                    st.markdown(line)
    elif section_type == "callout":
        body = format_doc_value(section.get("body", []), context)
        message = body if isinstance(body, str) else "\n\n".join(body)
        variant = section.get("variant", "info")
        getattr(st, variant, st.info)(message)
    elif section_type == "divider":
        st.divider()


def build_doc_context(
    summary: Dict[str, Any],
    registry: Dict[str, Any],
    base_rate: float,
) -> Dict[str, Any]:
    defaults = summary.get("defaults", {})
    numeric = summary.get("numeric_stats", {})
    context = {
        "base_rate_pct": f"{base_rate:.1%}",
        "base_rate_value": base_rate,
        "best_model": registry.get("best_model", "n/a"),
        "service_column_count": len(SERVICE_COLUMNS),
        "time_column_count": len(TIME_COLUMNS),
        "route_example": "JFK↔LAX",
        "age_default": int(defaults.get("Age", 35)),
        "customer_type_default": defaults.get("Customer Type", "Loyal"),
        "travel_type_default": defaults.get("Type of Travel", "Personal"),
        "class_default": defaults.get("Class", "Eco"),
        "distance_default": int(defaults.get("Flight Distance", 1000)),
        "arrival_delay_median": int(numeric.get("Arrival Delay in Minutes", {}).get("median", 5)),
        "departure_delay_median": int(numeric.get("Departure Delay in Minutes", {}).get("median", 5)),
    }
    return context


def render_documentation_page(
    registry: Dict[str, Any],
    summary: Dict[str, Any],
    base_rate: float,
    lang: str,
) -> None:
    doc_locale = get_doc_locale(lang)
    sections = doc_locale.get("sections", [])
    context = build_doc_context(summary, registry, base_rate)
    for section in sections:
        render_doc_section(section, context)

def render_wizard_page(
    registry: Dict[str, Any],
    models: Dict[str, Any],
    summary: Dict[str, Any],
    base_rate: float,
    lang: str,
    selected_model_name: str,
) -> None:
    st.header(translate(lang, "wizard.heading"))
    wizard_navigation(len(QUESTION_SECTIONS), lang)
    step = st.session_state.wizard_step
    current_section = QUESTION_SECTIONS[step]
    st.subheader(section_label(current_section, lang))
    description = section_description(current_section, lang)
    if description:
        st.caption(description)

    if current_section == "trip_basics":
        render_trip_basics(summary, lang)
    elif current_section == "experience":
        render_experience_section(summary, lang)
    else:
        render_ops_section(summary, lang)

    captured = st.session_state.responses
    with st.expander(translate(lang, "wizard.expander_label"), expanded=False):
        st.caption(translate(lang, "wizard.responses_help"))
        st.json(captured)

    if step == len(QUESTION_SECTIONS) - 1:
        if st.button(translate(lang, "wizard.generate_button"), type="primary", use_container_width=True):
            features = pd.DataFrame([captured])
            with st.spinner(translate(lang, "wizard.loading_message")):
                score_df = collect_model_scores(models, features, base_rate, lang)

            if score_df.empty:
                st.error(translate(lang, "messages.scoring_error"))
                st.stop()

            if selected_model_name not in score_df["model"].values:
                selected_model_name = score_df["model"].iat[0]
                st.session_state.selected_model = selected_model_name

            current_row = score_df[score_df["model"] == selected_model_name].iloc[0]
            probability = float(current_row["probability"])
            label = current_row["label"]
            emoji = current_row["emoji"]

            st.success(
                translate(
                    lang,
                    "wizard.score_message",
                    emoji=emoji,
                    model=selected_model_name,
                    probability=probability,
                    label=label,
                )
            )
            st.progress(probability)
            st.metric(
                translate(lang, "wizard.lift_label"),
                f"{probability - base_rate:+.1%}",
                help=translate(lang, "wizard.lift_help"),
            )

            severity, action_copy = next_best_action(probability, lang)
            getattr(st, severity)(f"{translate(lang, "wizard.nba_prefix")}: {action_copy}")

            st.write(f"#### {translate(lang, "wizard.agreement_heading")}")
            display_df = score_df.copy()
            display_df["probability"] = display_df["probability"].map(lambda x: f"{x:.1%}")
            display_df["lift"] = display_df["lift"].map(lambda x: f"{x:+.1%}")
            st.dataframe(
                display_df.set_index("model")[["probability", "lift", "label"]],
                use_container_width=True,
            )

            fig_scores = px.bar(
                score_df,
                x="probability",
                y="model",
                orientation="h",
                text="probability_label",
                range_x=[0, 1],
                title=translate(lang, "wizard.model_chart_title"),
                color="model",
            )
            fig_scores.update_layout(showlegend=False)
            st.plotly_chart(fig_scores, use_container_width=True)
            st.caption(translate(lang, "wizard.model_chart_help"))

            impacts = compute_personal_impacts(registry, captured, summary)
            if impacts:
                st.subheader(translate(lang, "wizard.drivers_title"))
                impact_df = pd.DataFrame(impacts)
                impact_fig = px.bar(
                    impact_df,
                    x="impact",
                    y="feature",
                    color="impact",
                    orientation="h",
                    title=translate(lang, "wizard.drivers_title"),
                    color_continuous_scale="Tealrose",
                )
                st.plotly_chart(impact_fig, use_container_width=True)
                with st.popover(translate(lang, "wizard.why_drivers")):
                    st.write(translate(lang, "wizard.drivers_explainer"))

            suggestions = build_suggestions(captured, impacts or [], probability, lang)
            st.write(f"### {translate(lang, "wizard.suggestions_heading")}")
            for tip in suggestions:
                st.write(f"- {tip}")


def render_analytics_page(registry: Dict[str, Any], lang: str) -> None:
    st.header(translate(lang, "analytics.title"))
    render_model_diagnostics(registry, lang)
    st.divider()
    render_global_signals(registry, lang)


summary = load_reference_summary()
ensure_session_state(summary)

if "language" not in st.session_state:
    st.session_state.language = DEFAULT_LANGUAGE
if "selected_page" not in st.session_state:
    st.session_state.selected_page = PAGE_CODES[0]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

lang = st.session_state.language
registry = load_registry()
if not registry:
    st.error(translate(lang, "messages.registry_missing"))
    st.stop()

models, missing_artifacts = load_models(registry)
for model_name, path in missing_artifacts:
    st.warning(translate(lang, "messages.artifact_missing", model=model_name, path=str(path)))

if not models:
    st.error(translate(lang, "messages.no_models"))
    st.stop()

model_names = list(models.keys())
if not model_names:
    st.error(translate(lang, "messages.no_models"))
    st.stop()

if st.session_state.selected_model not in model_names:
    st.session_state.selected_model = registry.get("best_model") or model_names[0]

base_rate = registry.get("base_rate", 0.5)

with st.sidebar:
    language_options = sorted(TRANSLATIONS.keys())
    current_language = (
        st.session_state.language if st.session_state.language in language_options else DEFAULT_LANGUAGE
    )
    st.selectbox(
        translate(lang, "sidebar.language_label"),
        language_options,
        index=language_options.index(current_language),
        format_func=language_label,
        key="language",
    )
    lang = st.session_state.language
    current_page = (
        st.session_state.selected_page if st.session_state.selected_page in PAGE_CODES else PAGE_CODES[0]
    )
    st.radio(
        translate(lang, "sidebar.navigation_label"),
        PAGE_CODES,
        index=PAGE_CODES.index(current_page),
        format_func=lambda code: page_label(code, lang),
        key="selected_page",
    )
    current_model = (
        st.session_state.selected_model if st.session_state.selected_model in model_names else model_names[0]
    )
    st.session_state.selected_model = current_model
    st.selectbox(
        translate(lang, "sidebar.model_selector"),
        model_names,
        index=model_names.index(current_model),
        format_func=lambda name: name.replace("_", " ").title(),
        help=translate(lang, "sidebar.model_selector_help"),
        key="selected_model",
    )
    st.metric(translate(lang, "sidebar.base_rate"), f"{base_rate:.2%}")
    st.info(
        f"{translate(lang, 'sidebar.best_model')}: **{registry.get('best_model', model_names[0])}**"
    )

st.title(translate(lang, "app.page_title"))
st.write(translate(lang, "app.tagline"))
st.caption(translate(lang, "app.wizard_only_note"))

if st.session_state.selected_page == "wizard":
    render_wizard_page(
        registry,
        models,
        summary,
        base_rate,
        lang,
        st.session_state.selected_model,
    )
elif st.session_state.selected_page == "analytics":
    render_analytics_page(registry, lang)
else:
    render_documentation_page(registry, summary, base_rate, lang)
