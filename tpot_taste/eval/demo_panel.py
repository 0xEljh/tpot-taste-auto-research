"""The demonstrative dipstick panel (Phase 6, D28) — a FIXED, versioned probe set.

A human-curated spectrum for *qualitative sensing* of the Scorer and Writer. Hand-authored archetypes
(stable, attribution-free, unambiguous category labels) that mirror the patterns the goods-audit surfaced:
genuine tpot, sharp aphorisms (the hard case vs platitude), motivational platitudes, promo/announcement
creep (the real leak, D28), bait, corporate, assistant-slop.

The dipstick test: does the Scorer rank EXPECTED_HIGH above EXPECTED_LOW? In particular, does it keep
`aphorism`/`tpot_canon` above `generic_viral`/`promo`? (D27/D28 say it blurs the aphorism↔platitude line.)

Re-run `scripts/demo_eval.py` after every model iteration and diff — this panel is the constant ruler.
"""
from __future__ import annotations

# category -> archetypes. Order of keys is the EXPECTED descending taste order.
PANEL: dict[str, list[str]] = {
    # --- should score HIGH ---
    "tpot_canon": [
        "spent 40 minutes optimizing a function that runs once a month. no regrets. the function and i "
        "both know what we are to each other now.",
        "the best code review i ever got was a single word — 'why?' — and it took me three days to admit "
        "i didn't have an answer.",
        "every config file is a little graveyard of past selves, each of whom swore that this time they'd "
        "keep it clean.",
        "i keep a text file of bugs that turned out to be my own misunderstanding. it is the most honest "
        "document i own.",
        "realized my entire aesthetic in software is just trying to make the thing that scared me this "
        "morning slightly less scary by tonight.",
    ],
    "aphorism": [
        "you don't find your taste. you notice it.",
        "most advice is autobiography in disguise.",
        "premature abstraction is just procrastination wearing a nicer coat.",
        "the trick to having good ideas is having far more bad ones and the nerve to tell them apart.",
        "the opposite of play isn't work, it's depression.",
    ],
    # --- should score LOW (these are the pollution we want the scorer to reject) ---
    "generic_viral": [  # motivational platitudes — work on broad Twitter, NOT tpot
        "discipline is choosing what you want most over what you want right now.",
        "you will never grow if you stay inside your comfort zone.",
        "success isn't about motivation, it's about consistency. show up every single day.",
        "every time you procrastinate you train your brain to avoid hard things. start now.",
        "the most important investment you will ever make is in yourself.",
    ],
    "promo": [  # announcement / launch creep — the real leak the audit found (D28)
        "Excited to announce the launch of our new community for builders! 🚀 Join us for support, "
        "feedback, and accountability. Link below 👇",
        "We're thrilled to share that our fellowship is now live — seeking talented researchers to push "
        "the frontier. Apply now! 🧠✨",
        "Big news: the v2 engine upgrade is live and already 45% of users have migrated. Try it free "
        "this week. 🎉",
    ],
    "corporate": [  # LinkedIn voice
        "Thrilled to share that after months of hard work our team shipped a feature that delivers real "
        "value to our customers. Grateful for this journey! 🙏 #blessed #teamwork",
        "Failure is just feedback. Here are 5 lessons from my biggest career setback — a thread 🧵 that "
        "will change how you think about success.",
    ],
    "bait": [  # engagement-bait — should be lowest
        "RT if you agree. like if you're a developer. reply 'yes' if you've ever shipped on a friday 😤🔥",
        "drop a 🔥 in the replies if you're building something right now. tag a friend who needs to see "
        "this 🙌",
        "who else is grinding tonight? 🙋 agree? let me know below 👇",
    ],
    "assistant_slop": [  # the base model's untrained register
        "Absolutely! Here's a great tweet for TPOT: 🚀 'Embracing the journey of continuous learning in "
        "tech!' Let me know if you'd like another version! 😊",
        "Sure! Here are some engaging post ideas about coding: 1) Stay curious! 2) Keep building! "
        "3) Never stop learning! #coding #motivation",
    ],
}

EXPECTED_HIGH = ["tpot_canon", "aphorism"]
EXPECTED_LOW = ["generic_viral", "promo", "corporate", "bait", "assistant_slop"]

# Writer panel: fixed topics (ideate) + fixed drafts (improve). `discipline` and the platitude draft are
# deliberate traps — does the Writer drift to motivational genericness, or stay tpot-specific?
IDEATE_TOPICS = [
    "recursion",
    "debugging at 2am",
    "the feeling of deleting a lot of code",
    "discipline",  # platitude-bait topic
    "reading philosophy you don't fully understand",
    "your desk",
]

IMPROVE_DRAFTS = [
    "lit a fake cig to feel something",  # the benchmark smoke-test
    "work hard and you will succeed",  # platitude draft — does `improve` de-platitude or amplify?
    "i think therefore i am but for code",
]
