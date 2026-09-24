---

## Project 3: AI Recommendation Logic (Tech Stack Matchmaker)
**Focus:** Content-Based Filtering, Vector Space Modeling, and Cosine Similarity

### Overview
An intelligent matchmaking engine that ingests user skills and career interests, converts unstructured metadata into TF-IDF vector representations, and ranks the Top 3 relevant career paths using angular Cosine Similarity.

### Key Features
- **4-Step Engine Pipeline:** Ingestion, TF-IDF Scoring, Descending Sorting, and Top-N Filtering
- **Similarity Metric:** Cosine Similarity invariant to vector magnitude
- **Dataset:** `raw_skills.csv` mapping industry roles to tool stacks
- **Bypasses:** Handles Cold Start scenarios with baseline profile defaults
