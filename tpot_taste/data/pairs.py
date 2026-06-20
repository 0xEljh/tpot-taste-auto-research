"""Mine same-author preference pairs (the Scorer's primary, confound-robust signal).

Within (author, era), pair the highest-z tweets against the lowest-z tweets, keeping
only pairs with a clear engagement margin (ratio + absolute + z-gap). Same author and
era difference out the follower/era confounds (Tan-Lee-Pang 2014; PopALM top-vs-less).
Bounded work: only top-k vs bottom-k per cell, capped per author so prolific authors
(eigenrobot, visakanv) don't dominate.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def mine_pairs(
    df: pl.DataFrame,
    *,
    author_col: str = "account_id",
    era_col: str = "era",
    fav_col: str = "favorite_count",
    z_col: str = "z",
    text_col: str = "full_text",
    id_col: str = "tweet_id",
    min_ratio: float = 2.0,
    min_abs_gap: float = 5.0,
    min_z_gap: float = 0.75,
    min_winner_fav: int = 0,
    max_pairs_per_author: int = 50,
    same_reply_class: bool = True,
) -> pl.DataFrame:
    rows: list[dict] = []

    for adf in df.partition_by(author_col, maintain_order=True):
        aid = adf[author_col][0]
        made = 0
        for edf in adf.partition_by(era_col, maintain_order=True):
            if made >= max_pairs_per_author or edf.height < 2:
                continue
            era = edf[era_col][0]
            s = edf.sort(z_col, descending=True, maintain_order=True)
            k = min(s.height // 2, max_pairs_per_author - made)
            if k <= 0:
                continue
            winners = list(s.head(k).iter_rows(named=True))
            losers = list(s.tail(k).iter_rows(named=True))[::-1]  # worst aligned with best
            has_reply = "is_reply" in edf.columns
            for w, l in zip(winners, losers):
                if made >= max_pairs_per_author:
                    break
                if w[text_col] == l[text_col]:  # repost vs itself — not a quality contrast
                    continue
                fw, fl = w[fav_col], l[fav_col]
                if fw < min_winner_fav:
                    continue
                if fw <= fl:
                    continue
                if fl > 0 and fw < min_ratio * fl:
                    continue
                if (fw - fl) < min_abs_gap:
                    continue
                if (w[z_col] - l[z_col]) < min_z_gap:
                    continue
                if same_reply_class and has_reply and w.get("is_reply") != l.get("is_reply"):
                    continue
                rows.append(
                    {
                        f"{author_col}_w": aid,
                        f"{author_col}_l": aid,
                        "era": era,
                        f"{id_col}_w": w[id_col],
                        f"{text_col}_w": w[text_col],
                        f"{fav_col}_w": fw,
                        f"{z_col}_w": w[z_col],
                        f"{id_col}_l": l[id_col],
                        f"{text_col}_l": l[text_col],
                        f"{fav_col}_l": fl,
                        f"{z_col}_l": l[z_col],
                    }
                )
                made += 1

    return pl.DataFrame(rows)


def mine_topic_pairs(
    df: pl.DataFrame,
    emb: np.ndarray,
    *,
    author_col: str = "account_id",
    era_col: str = "era",
    fav_col: str = "favorite_count",
    z_col: str = "z",
    text_col: str = "text_clean",
    id_col: str = "tweet_id",
    min_ratio: float = 2.0,
    min_abs_gap: float = 5.0,
    min_z_gap: float = 0.75,
    min_winner_fav: int = 10,
    min_topic_cos: float = 0.5,
    max_pairs_per_author: int = 150,
    winners_frac: float = 0.5,
) -> pl.DataFrame:
    """Same-author pairs that are ALSO topically matched (Scorer v2, D12).

    `emb` is an L2-normalized matrix aligned row-for-row with `df`. Within each author,
    high-z winners are greedily matched to the most *topically-similar* low-z loser
    (cosine ≥ min_topic_cos) that still satisfies the engagement margins — so the pair
    differs in craft, not subject.
    """
    emb = np.asarray(emb, dtype=np.float32)
    favs = df[fav_col].to_numpy()
    zs = df[z_col].to_numpy()
    ids = df[id_col].to_list()
    texts = df[text_col].to_list()
    eras = df[era_col].to_list() if era_col in df.columns else [0] * df.height
    authors = df[author_col].to_list()

    groups = df.with_row_index("__r").group_by(author_col).agg(pl.col("__r"))
    rows: list[dict] = []

    for grp in groups.iter_rows(named=True):
        ridx = np.asarray(grp["__r"])
        if ridx.size < 2:
            continue
        order = ridx[np.argsort(-zs[ridx])]
        half = max(1, int(order.size * winners_frac))
        winners, losers = order[:half], order[half:]
        if losers.size == 0:
            continue

        cos = emb[winners] @ emb[losers].T  # nw × nl
        fw, fl = favs[winners][:, None], favs[losers][None, :]
        zw, zl = zs[winners][:, None], zs[losers][None, :]
        valid = (
            (fw > fl)
            & ((fl == 0) | (fw >= min_ratio * fl))
            & ((fw - fl) >= min_abs_gap)
            & ((zw - zl) >= min_z_gap)
            & (fw >= min_winner_fav)
            & (cos >= min_topic_cos)
            & (cos < 0.995)  # exclude near-duplicate reposts
        )
        score = np.where(valid, cos, -1.0)
        used = np.zeros(losers.size, dtype=bool)
        aid = authors[int(winners[0])]
        made = 0
        for wi in range(winners.size):
            if made >= max_pairs_per_author:
                break
            row = score[wi].copy()
            row[used] = -1.0
            j = int(np.argmax(row))
            if row[j] < min_topic_cos:
                continue
            used[j] = True
            w, l = int(winners[wi]), int(losers[j])
            rows.append({
                f"{author_col}_w": aid, f"{author_col}_l": aid, "era": eras[w],
                f"{id_col}_w": ids[w], f"{text_col}_w": texts[w],
                f"{fav_col}_w": int(favs[w]), f"{z_col}_w": float(zs[w]),
                f"{id_col}_l": ids[l], f"{text_col}_l": texts[l],
                f"{fav_col}_l": int(favs[l]), f"{z_col}_l": float(zs[l]),
                "topic_cos": float(cos[wi, j]),
            })
            made += 1

    return pl.DataFrame(rows)
