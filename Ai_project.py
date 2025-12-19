"""
AI-Based Political Investability Scoring System (0–10)
===================================================


DISCLAIMER:
- This is an educational / research prototype.
- Scores are illustrative, NOT financial advice.
- Focus: architecture, reasoning, explainability.
"""

# =========================
# IMPORTS
# =========================
import requests
import pandas as pd
from datetime import datetime
import math

# Optional NLP (can be disabled if hardware is weak)
try:
    from transformers import pipeline
    NLP_AVAILABLE = True
except ImportError:
    pipeline = None
    NLP_AVAILABLE = False

# =========================
# CONFIGURATION
# =========================
COUNTRY = "Turkey"
NEWS_API_KEY = "YOUR_NEWSAPI_KEY"  # <-- replace
NEWS_LOOKBACK_DAYS = 30
UPDATE_INTERVAL_MINUTES = 60

# Baseline political stability (neutral = 5)
BASELINE_SCORE = 5.0

# =========================
# EVENT DEFINITIONS
# =========================
EVENT_WEIGHTS = {
    "Stable Election": 1.2,
    "Policy Reform": 0.8,
    "Peace Agreement": 1.5,
    "Minor Protest": -0.4,
    "Mass Protest": -1.2,
    "Corruption Scandal": -1.5,
    "Sanctions": -1.8,
    "Coup / Armed Conflict": -2.5
}

SOURCE_CREDIBILITY = {
    "reuters": 1.0,
    "bbc": 0.95,
    "ap": 0.95,
    "al jazeera": 0.9,
    "local": 0.8
}

# =========================
# NLP MODELS
# =========================
if NLP_AVAILABLE and pipeline is not None:
    relevance_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    event_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
else:
    relevance_classifier = None
    event_classifier = None

EVENT_LABELS = list(EVENT_WEIGHTS.keys())

# =========================
# NEWS INGESTION
# =========================
def fetch_latest_news():
    """Fetch latest news articles for the selected country."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": COUNTRY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50,
        "apiKey": ""
    }

    response = requests.get(url, params=params)
    data = response.json()

    articles = []
    for item in data.get("articles", []):
        articles.append({
            "title": item["title"],
            "text": item["description"] or "",
            "source": item["source"]["name"].lower(),
            "published": datetime.fromisoformat(item["publishedAt"].replace("Z", ""))
        })

    return pd.DataFrame(articles)

# =========================
# POLITICAL RELEVANCE FILTER
# =========================
def is_politically_relevant(text):
    keywords = ["election", "government", "policy", "protest", "sanction", "war", "corruption"]
    if any(k in text.lower() for k in keywords):
        return True

    if NLP_AVAILABLE:
        result = relevance_classifier(
            text,
            candidate_labels=["politics", "sports", "entertainment", "technology"],
        )
        return result["labels"][0] == "politics"

    return False

# =========================
# EVENT CLASSIFICATION
# =========================
def classify_event(text):
    if NLP_AVAILABLE:
        result = event_classifier(text, EVENT_LABELS)
        return result["labels"][0], result["scores"][0]

    # fallback rule-based
    text = text.lower()
    if "coup" in text or "armed" in text:
        return "Coup / Armed Conflict", 0.9
    if "sanction" in text:
        return "Sanctions", 0.9
    if "protest" in text:
        return "Mass Protest", 0.7
    if "corruption" in text:
        return "Corruption Scandal", 0.8
    if "election" in text:
        return "Stable Election", 0.6

    return "Policy Reform", 0.5

# =========================
# SCORING ENGINE
# =========================
def time_decay(event_date):
    days_old = (datetime.now() - event_date).days
    return math.exp(-days_old / 14)


def credibility_weight(source):
    for key in SOURCE_CREDIBILITY:
        if key in source:
            return SOURCE_CREDIBILITY[key]
    return SOURCE_CREDIBILITY["local"]


def compute_event_score(event_type, confidence, source, published):
    base = EVENT_WEIGHTS.get(event_type, 0)
    credibility = credibility_weight(source)
    decay = time_decay(published)

    return base * confidence * credibility * decay

# =========================
# AGGREGATION
# =========================
def compute_final_score(events_df):
    total_impact = events_df["event_score"].sum()
    volatility = events_df["event_score"].std() if len(events_df) > 1 else 0

    score = BASELINE_SCORE + total_impact - volatility
    return max(0, min(10, score))

# =========================
# EXPLAINABILITY
# =========================
def generate_explanation(events_df, final_score):
    top_events = events_df.sort_values("event_score").head(3)

    explanation = f"Final political investability score for {COUNTRY}: {final_score:.2f}\n"
    explanation += "Main risk drivers:\n"

    for _, row in top_events.iterrows():
        explanation += f"- {row['event_type']} ({row['event_score']:.2f})\n"

    return explanation

# =========================
# MAIN PIPELINE
# =========================
def run_pipeline():
    news = fetch_latest_news()

    relevant_articles = []
    for _, row in news.iterrows():
        if is_politically_relevant(row["title"] + " " + row["text"]):
            event_type, confidence = classify_event(row["title"] + " " + row["text"])
            score = compute_event_score(
                event_type,
                confidence,
                row["source"],
                row["published"]
            )
            relevant_articles.append({
                "event_type": event_type,
                "confidence": confidence,
                "event_score": score
            })

    events_df = pd.DataFrame(relevant_articles)

    if events_df.empty:
        return BASELINE_SCORE, "No significant political events detected."

    final_score = compute_final_score(events_df)
    explanation = generate_explanation(events_df, final_score)

    return final_score, explanation


# =========================
# RUN
# =========================
if __name__ == "__main__":
    score, explanation = run_pipeline()
    print(explanation)
