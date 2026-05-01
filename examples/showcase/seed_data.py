"""Labeled paraphrase corpus for the showcase.

Each entry: ``(query, true_intent)``. Designed with deliberate paraphrase
clusters so the stress test produces a visible cache-warm-up curve.
"""

from __future__ import annotations

import random

# 7 intents x ~10-12 paraphrases = ~80 messages. Multiple paraphrases per
# intent ensure the semantic-match layer earns its keep on the stress run.
MESSAGES: list[tuple[str, str]] = [
    # --- billing ---
    ("My credit card was charged twice for the same order.", "billing"),
    ("I see a duplicate charge on my statement.", "billing"),
    ("Why was I billed twice this month?", "billing"),
    ("There's a strange charge on my account I don't recognize.", "billing"),
    ("How do I update my billing address?", "billing"),
    ("I need to change the credit card on file.", "billing"),
    ("Where do I update payment information?", "billing"),
    ("Can I get an itemized receipt for last month?", "billing"),
    ("Send me a copy of my invoice please.", "billing"),
    ("How do I download my past invoices?", "billing"),
    ("My subscription renewed but I wanted to cancel first.", "billing"),
    # --- technical ---
    ("The app keeps crashing whenever I try to log in.", "technical"),
    ("Login causes the application to crash on iPhone.", "technical"),
    ("App crashes on launch since the latest update.", "technical"),
    ("The page won't load, I just see a spinning wheel.", "technical"),
    ("Getting a 500 error when I submit the form.", "technical"),
    ("Why am I seeing 'Internal Server Error'?", "technical"),
    ("The dashboard shows blank data.", "technical"),
    ("Charts aren't rendering on the analytics page.", "technical"),
    ("My uploads keep failing midway.", "technical"),
    ("Files won't upload, they fail at 50%.", "technical"),
    ("The mobile app won't sync with the web version.", "technical"),
    # --- refund ---
    ("I want a refund for my order.", "refund"),
    ("How can I get my money back?", "refund"),
    ("Please refund the charge from yesterday.", "refund"),
    ("Can I return this and get my money back?", "refund"),
    ("What's your refund policy?", "refund"),
    ("How long does a refund take to process?", "refund"),
    ("When will I see my refund hit my card?", "refund"),
    ("My refund was supposed to arrive last week.", "refund"),
    ("I was promised a refund but never got it.", "refund"),
    ("Where is my refund? It's been 14 days.", "refund"),
    # --- account ---
    ("How do I reset my password?", "account"),
    ("I forgot my password, what do I do?", "account"),
    ("Password reset isn't working for me.", "account"),
    ("Help, I can't log in.", "account"),
    ("My account is locked out.", "account"),
    ("How do I change my email address?", "account"),
    ("I need to update my email on file.", "account"),
    ("How do I delete my account?", "account"),
    ("I want to permanently close my account.", "account"),
    ("Can I merge two accounts I have?", "account"),
    ("How do I enable two-factor authentication?", "account"),
    # --- how_to ---
    ("How do I export my data to CSV?", "how_to"),
    ("Where can I download all my records as a spreadsheet?", "how_to"),
    ("How do I share a project with a teammate?", "how_to"),
    ("Where is the option to invite collaborators?", "how_to"),
    ("How do I set up notifications?", "how_to"),
    ("Can I configure email alerts for new activity?", "how_to"),
    ("How do I change the language to Spanish?", "how_to"),
    ("Where do I switch the interface language?", "how_to"),
    ("How do I integrate with Slack?", "how_to"),
    ("Is there a way to connect this to Microsoft Teams?", "how_to"),
    # --- complaint ---
    ("Your support has been terrible.", "complaint"),
    ("I'm extremely frustrated with the service.", "complaint"),
    ("This is the worst customer experience I've ever had.", "complaint"),
    ("Nobody has gotten back to me in 5 days.", "complaint"),
    ("I've been on hold for an hour, this is unacceptable.", "complaint"),
    ("Your product is full of bugs and getting worse.", "complaint"),
    ("I'm canceling and telling everyone how bad this is.", "complaint"),
    ("This is ridiculous, I want to speak to a manager.", "complaint"),
    ("Why is your service so unreliable?", "complaint"),
    ("I'm done with your company.", "complaint"),
    # --- other (chitchat / off-topic / unclassifiable) ---
    ("Hello, are you there?", "other"),
    ("Hi.", "other"),
    ("Just testing if anyone reads these.", "other"),
    ("What's the weather like?", "other"),
    ("Who is your CEO?", "other"),
    ("Tell me a joke.", "other"),
    ("Are you a human or a bot?", "other"),
    ("What time do you close today?", "other"),
    ("Where are your offices located?", "other"),
    ("Can I get a job at your company?", "other"),
]


# Click-to-fill presets on the Try-it page: one per intent, the most-iconic example.
TRY_IT_PRESETS: list[str] = [
    "How do I reset my password?",
    "My credit card was charged twice for the same order.",
    "The app keeps crashing whenever I try to log in.",
    "I want a refund for my order.",
    "How do I export my data to CSV?",
    "Your support has been terrible.",
    "Tell me a joke.",
]


def stress_run_order(seed: int = 1) -> list[tuple[str, str]]:
    """Shuffle the corpus into a deterministic run order for the stress test."""
    rng = random.Random(seed)
    out = list(MESSAGES)
    rng.shuffle(out)
    return out


# Paraphrase pairs for the calibration script. Built from clusters above where
# multiple queries share the same intent — a paraphrase pair is two queries
# from the same intent cluster.
def paraphrase_pairs() -> list[tuple[str, str]]:
    by_intent: dict[str, list[str]] = {}
    for q, intent in MESSAGES:
        by_intent.setdefault(intent, []).append(q)
    pairs: list[tuple[str, str]] = []
    for queries in by_intent.values():
        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                pairs.append((queries[i], queries[j]))
    return pairs


def distractor_pairs() -> list[tuple[str, str]]:
    """All cross-intent pairs (different intents -> should NOT match)."""
    pairs: list[tuple[str, str]] = []
    for i, (q_i, intent_i) in enumerate(MESSAGES):
        for q_j, intent_j in MESSAGES[i + 1 :]:
            if intent_i != intent_j:
                pairs.append((q_i, q_j))
    return pairs
