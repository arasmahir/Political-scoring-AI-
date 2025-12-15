# Political Investability Scoring AI

This project is a Python-based system that analyzes real-time political news
to generate an investability score (0–10) for a given country based on
political stability and risk.

## How It Works
- Fetches live news using NewsAPI
- Detects politically relevant events using NLP
- Classifies events (protests, reforms, conflicts, etc.)
- Assigns weighted impacts to events
- Aggregates results into a single explainable score

## Example Output


## Setup
1. Clone the repository
2. Install dependencies:

pip install -r requirements.txt

3. Set your NewsAPI key:

export NEWS_API_KEY=your_key_here

4. Run:

This is a v1 prototype focused on correctness and explainability.
Future improvements include event deduplication, time decay, and confidence intervals.

WARNING THIS IS FOR EDUCATIONAL PURPOSES ONLY DO NOT TAKE THIS AI PROGRAM AS AN INVESTMENT ADVISE
