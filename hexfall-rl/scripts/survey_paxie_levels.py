"""
scripts/survey_paxie_levels.py

Surveys all level JSONs in CLASSIFIED.paxie_data/level_data/ and emits an
aggregate-only markdown report to CLASSIFIED.paxie_data/survey_report.md.

SECURITY RULES (hardcoded, not configurable):
  - Never prints full level contents, stack contents, tunnel queues, or cell lists.
  - Aggregates only: frequencies, distributions, inventories of keys/values.
  - No per-level detail beyond level-number labelling of anomalies and
    per-bucket difficulty-progression summaries.

Usage:
    python scripts/survey_paxie_levels.py
    python scripts/survey_paxie_levels.py --data-dir path/to/CLASSIFIED.paxie_data/level_data
    python scripts/survey_paxie_levels.py --out path/to/report.md
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_load(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"
    except OSError as e:
        return None, f"OSError: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def _level_num(filename: str):
    m = re.search(r"(\d+)", Path(filename).stem)
    return int(m.group(1)) if m else None


def _stats(values):
    if not values:
        return {"count": 0}
    d = {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 2),
        "median": median(values),
    }
    if len(values) > 1:
        try:
            d["stdev"] = round(stdev(values), 2)
        except Exception:
            pass
    return d


def _histogram(values, buckets):
    hist = defaultdict(int)
    for v in values:
        placed = False
        for b in buckets:
            if v <= b:
                hist[f"<={b}"] += 1
                placed = True
                break
        if not placed:
            hist[f">{buckets[-1]}"] += 1
    # Preserve bucket order
    ordered = {}
    for b in buckets:
        key = f"<={b}"
        if key in hist:
            ordered[key] = hist[key]
    overflow = f">{buckets[-1]}"
    if overflow in hist:
        ordered[overflow] = hist[overflow]
    return ordered


def _is_color_code(v) -> bool:
    """Recognise Paxie color tokens: short alphabetic codes (`y`, `dg`, `br`,
    `dgr`) plus longer human-readable names (`Green`, `DarkRed`, `OliveGreen`).
    """
    if not isinstance(v, str):
        return False
    if not v:
        return False
    # Permit only ASCII letters
    return all(c.isalpha() for c in v) and len(v) <= 20


def _collect_item_keys(items):
    keys = set()
    for it in items or []:
        if isinstance(it, dict):
            keys.update(it.keys())
    return sorted(keys)


# ---------------------------------------------------------------------------
# Known schema constants for THIS dataset (Paxie Hex Fall 100-level export)
# ---------------------------------------------------------------------------

CA_MECHANIC_ARRAYS = (
    "singleBlockCollectors",
    "iceCollectors",
    "pinBlockers",
    "mysteryCollectors",
    "woodBoxCollectors",
    "tunnels",
    "deadCells",
    "keyLocks",
    "tiedPairs",
)

# editorMeta fields we know how to handle and want stats for
EM_NUMERIC_FIELDS = (
    "totalBlocks",
    "colorCount",
    "maxColorsPerStack",
    "heightMin",
    "heightMax",
    "randomness",
    "verticalPercent",
    "horizontalPercent",
    "mysteryPercent",
    "clusteringPercent",
    "zoneCount",
)
EM_BOOL_FIELDS = ("clusteringEnabled",)


# ---------------------------------------------------------------------------
# Per-level extraction
# Returns ONLY aggregate-safe scalars, key names, and per-mechanic counts.
# ---------------------------------------------------------------------------

def _extract_collector_area(data: dict) -> dict:
    ca = data.get("collectorArea")
    result = {"present": isinstance(ca, dict)}
    if not isinstance(ca, dict):
        return result
    result["keys"] = sorted(ca.keys())
    result["gridWidth"] = ca.get("gridWidth")
    result["gridHeight"] = ca.get("gridHeight")

    # Mechanic counts + item-level key inventories
    for arr_key in CA_MECHANIC_ARRAYS:
        items = ca.get(arr_key)
        if isinstance(items, list):
            result[f"{arr_key}_count"] = len(items)
            result[f"{arr_key}_item_keys"] = _collect_item_keys(items)

    # Specific numeric/categorical values for ice / pin / tunnel
    ice_caps = []
    for it in ca.get("iceCollectors", []) or []:
        if isinstance(it, dict):
            cap = it.get("iceCapacity")
            if isinstance(cap, (int, float)):
                ice_caps.append(cap)
    result["iceCapacities"] = ice_caps

    pin_block_counts = []
    pin_directions = []
    for it in ca.get("pinBlockers", []) or []:
        if isinstance(it, dict):
            bc = it.get("blockCount")
            if isinstance(bc, (int, float)):
                pin_block_counts.append(bc)
            d = it.get("direction")
            if isinstance(d, str):
                pin_directions.append(d)
    result["pinBlockCounts"] = pin_block_counts
    result["pinDirections"] = pin_directions

    tunnel_directions = []
    tunnel_queue_lengths = []
    tunnel_queue_colors = []
    for t in ca.get("tunnels", []) or []:
        if isinstance(t, dict):
            d = t.get("direction")
            if isinstance(d, str):
                tunnel_directions.append(d)
            q = t.get("collectorQueue")
            if isinstance(q, list):
                tunnel_queue_lengths.append(len(q))
                for item in q:
                    if isinstance(item, dict):
                        c = item.get("color")
                        if isinstance(c, str):
                            tunnel_queue_colors.append(c)
                    elif isinstance(item, str):
                        tunnel_queue_colors.append(item)
    result["tunnelDirections"] = tunnel_directions
    result["tunnelQueueLengths"] = tunnel_queue_lengths
    result["tunnelQueueColors"] = tunnel_queue_colors

    return result


def _extract_hex_stack_area(data: dict) -> dict:
    hsa = data.get("hexStackArea")
    result = {"present": isinstance(hsa, dict)}
    if not isinstance(hsa, dict):
        return result
    result["keys"] = sorted(hsa.keys())
    result["gridWidth"] = hsa.get("gridWidth")
    result["gridHeight"] = hsa.get("gridHeight")

    stacks = hsa.get("stacks")
    stack_item_keys = set()
    heights = []
    colors_in_stacks = []
    if isinstance(stacks, list):
        result["stack_count"] = len(stacks)
        for s in stacks:
            if isinstance(s, dict):
                stack_item_keys.update(s.keys())
                cols = s.get("colors")
                if isinstance(cols, list):
                    heights.append(len(cols))
                    for c in cols:
                        if isinstance(c, str):
                            colors_in_stacks.append(c)
    else:
        result["stack_count"] = 0
    result["stack_item_keys"] = sorted(stack_item_keys)
    result["stack_heights"] = heights
    result["colors_in_stacks"] = colors_in_stacks

    # Tunnels in HSA (observed empty in this dataset but check anyway)
    tunnels = hsa.get("tunnels") or []
    if isinstance(tunnels, list):
        result["tunnel_count"] = len(tunnels)
        result["tunnel_item_keys"] = _collect_item_keys(tunnels)
        tunnel_directions = []
        tunnel_queue_lengths = []
        tunnel_queue_colors = []
        for t in tunnels:
            if isinstance(t, dict):
                d = t.get("direction")
                if isinstance(d, str):
                    tunnel_directions.append(d)
                q = t.get("collectorQueue")
                if isinstance(q, list):
                    tunnel_queue_lengths.append(len(q))
                    for item in q:
                        if isinstance(item, dict):
                            c = item.get("color")
                            if isinstance(c, str):
                                tunnel_queue_colors.append(c)
        result["tunnelDirections"] = tunnel_directions
        result["tunnelQueueLengths"] = tunnel_queue_lengths
        result["tunnelQueueColors"] = tunnel_queue_colors

    return result


def _extract_editor_meta(data: dict) -> dict:
    em = data.get("editorMeta")
    result = {"present": isinstance(em, dict)}
    if not isinstance(em, dict):
        return result
    result["keys"] = sorted(em.keys())
    for f in EM_NUMERIC_FIELDS:
        if f in em:
            result[f] = em[f]
    for f in EM_BOOL_FIELDS:
        if f in em:
            result[f] = em[f]

    # colorsPerStackRatios: list of numbers — capture lengths + per-entry values
    cpsr = em.get("colorsPerStackRatios")
    if isinstance(cpsr, list):
        result["colorsPerStackRatios_len"] = len(cpsr)
        result["colorsPerStackRatios_values"] = [
            v for v in cpsr if isinstance(v, (int, float))
        ]

    # zoneColors: list of list-of-colors per zone (or list of color strings)
    zc = em.get("zoneColors")
    if isinstance(zc, list):
        result["zoneColors_len"] = len(zc)
        if zc and all(isinstance(x, list) for x in zc):
            result["zoneColors_type"] = "list_of_lists"
            result["zoneColors_inner_lens"] = [len(x) for x in zc]
            # Flatten inner color codes for inventory aggregation
            flat = []
            for inner in zc:
                for v in inner:
                    if isinstance(v, str):
                        flat.append(v)
            result["zoneColors_flat"] = flat
        elif zc and all(isinstance(x, str) for x in zc):
            result["zoneColors_type"] = "list_of_strings"
            result["zoneColors_flat"] = zc
        else:
            result["zoneColors_type"] = "mixed_or_other"

    return result


def _all_colors_in_level(data: dict) -> set:
    """Collect every distinct color token referenced anywhere in the level
    (stacks, singleBlockCollectors, ice/wood hidden colors, tunnel queues,
    keyLocks, zoneColors).
    """
    colors = set()

    hsa = data.get("hexStackArea") or {}
    if isinstance(hsa, dict):
        for s in hsa.get("stacks", []) or []:
            if isinstance(s, dict):
                for c in s.get("colors", []) or []:
                    if _is_color_code(c):
                        colors.add(c)
        for t in hsa.get("tunnels", []) or []:
            if isinstance(t, dict):
                for item in t.get("collectorQueue", []) or []:
                    if isinstance(item, dict):
                        c = item.get("color")
                        if _is_color_code(c):
                            colors.add(c)

    ca = data.get("collectorArea") or {}
    if isinstance(ca, dict):
        for sbc in ca.get("singleBlockCollectors", []) or []:
            if isinstance(sbc, dict):
                c = sbc.get("color")
                if _is_color_code(c):
                    colors.add(c)
        for it in ca.get("iceCollectors", []) or []:
            if isinstance(it, dict):
                c = it.get("hiddenColor")
                if _is_color_code(c):
                    colors.add(c)
        for it in ca.get("woodBoxCollectors", []) or []:
            if isinstance(it, dict):
                c = it.get("hiddenColor")
                if _is_color_code(c):
                    colors.add(c)
        for it in ca.get("keyLocks", []) or []:
            if isinstance(it, dict):
                c = it.get("color")
                if _is_color_code(c):
                    colors.add(c)
        for t in ca.get("tunnels", []) or []:
            if isinstance(t, dict):
                for item in t.get("collectorQueue", []) or []:
                    if isinstance(item, dict):
                        c = item.get("color")
                        if _is_color_code(c):
                            colors.add(c)
                    elif _is_color_code(item):
                        colors.add(item)

    em = data.get("editorMeta") or {}
    if isinstance(em, dict):
        zc = em.get("zoneColors")
        if isinstance(zc, list):
            for entry in zc:
                if isinstance(entry, list):
                    for v in entry:
                        if _is_color_code(v):
                            colors.add(v)
                elif _is_color_code(entry):
                    colors.add(entry)

    return colors


# ---------------------------------------------------------------------------
# Main survey
# ---------------------------------------------------------------------------

def survey(data_dir: Path) -> dict:
    files = sorted(data_dir.glob("*.json"))
    if not files:
        print(f"WARNING: No JSON files found in {data_dir}", file=sys.stderr)

    malformed = []
    loaded_levels = []  # list of (level_num, parsed_data, filename)
    level_nums_seen = []

    # Aggregate containers
    top_level_key_counts = Counter()
    ca_key_counts = Counter()
    hsa_key_counts = Counter()
    em_key_counts = Counter()

    # Top-level scalar values (levelNumber, levelVersionCode)
    top_levelNumber_vals = []
    top_levelVersionCode_vals = []

    # CA / HSA grid dims
    ca_gridW, ca_gridH = [], []
    hsa_gridW, hsa_gridH = [], []

    # Per-mechanic stats
    mech_count_per_level = {arr: [] for arr in CA_MECHANIC_ARRAYS}
    mech_levels_with = {arr: [] for arr in CA_MECHANIC_ARRAYS}
    mech_item_keys = {arr: set() for arr in CA_MECHANIC_ARRAYS}

    # Stack stats
    hsa_stack_counts = []
    hsa_stack_heights = []
    hsa_stack_item_keys = set()

    # HSA tunnel stats
    hsa_tunnel_counts = []
    hsa_tunnel_levels = []
    hsa_tunnel_directions = []
    hsa_tunnel_queue_lengths = []
    hsa_tunnel_queue_colors = []
    hsa_tunnel_item_keys = set()

    # CA tunnel stats
    ca_tunnel_directions = []
    ca_tunnel_queue_lengths = []
    ca_tunnel_queue_colors = []

    # Ice / pin specifics
    ice_capacities = []
    pin_block_counts = []
    pin_directions = []

    # Color inventory
    global_color_counter = Counter()  # # levels containing each color
    per_level_color_count = []
    per_level_color_sets = {}  # level_num -> set
    color_consistency_issues = []

    # editorMeta numeric fields
    em_numeric_vals = defaultdict(list)
    em_bool_vals = defaultdict(Counter)
    em_zoneColors_types = Counter()
    em_zoneColors_lens = []
    em_zoneColors_inner_lens = []
    em_colorsPerStackRatios_lens = []
    em_colorsPerStackRatios_values = []

    # Schema baseline & anomalies
    level50_top_keys = None
    schema_surprises = []

    # --- First pass: load every file, accumulate stats ----------------------
    for path in files:
        data, err = _safe_load(path)
        if err:
            malformed.append((path.name, err))
            continue
        if not isinstance(data, dict):
            malformed.append((path.name, "Top-level JSON is not an object"))
            continue

        level_num = _level_num(path.name)
        if level_num is not None:
            level_nums_seen.append(level_num)
        loaded_levels.append((level_num, data, path.name))

        for k in data.keys():
            top_level_key_counts[k] += 1

        ln = data.get("levelNumber")
        if isinstance(ln, (int, float)):
            top_levelNumber_vals.append(ln)
        lvc = data.get("levelVersionCode")
        if isinstance(lvc, (int, float)):
            top_levelVersionCode_vals.append(lvc)

        if level_num == 50:
            level50_top_keys = set(data.keys())

        # CollectorArea
        ca = _extract_collector_area(data)
        if ca["present"]:
            for k in ca["keys"]:
                ca_key_counts[k] += 1
            if isinstance(ca.get("gridWidth"), (int, float)):
                ca_gridW.append(ca["gridWidth"])
            if isinstance(ca.get("gridHeight"), (int, float)):
                ca_gridH.append(ca["gridHeight"])
            for arr in CA_MECHANIC_ARRAYS:
                cnt = ca.get(f"{arr}_count", 0)
                mech_count_per_level[arr].append(cnt)
                if cnt > 0 and level_num is not None:
                    mech_levels_with[arr].append(level_num)
                keys = ca.get(f"{arr}_item_keys", [])
                mech_item_keys[arr].update(keys)
            ice_capacities.extend(ca.get("iceCapacities", []))
            pin_block_counts.extend(ca.get("pinBlockCounts", []))
            pin_directions.extend(ca.get("pinDirections", []))
            ca_tunnel_directions.extend(ca.get("tunnelDirections", []))
            ca_tunnel_queue_lengths.extend(ca.get("tunnelQueueLengths", []))
            ca_tunnel_queue_colors.extend(ca.get("tunnelQueueColors", []))

        # HexStackArea
        hsa = _extract_hex_stack_area(data)
        if hsa["present"]:
            for k in hsa["keys"]:
                hsa_key_counts[k] += 1
            if isinstance(hsa.get("gridWidth"), (int, float)):
                hsa_gridW.append(hsa["gridWidth"])
            if isinstance(hsa.get("gridHeight"), (int, float)):
                hsa_gridH.append(hsa["gridHeight"])
            hsa_stack_counts.append(hsa.get("stack_count", 0))
            hsa_stack_heights.extend(hsa.get("stack_heights", []))
            hsa_stack_item_keys.update(hsa.get("stack_item_keys", []))
            tc = hsa.get("tunnel_count", 0)
            hsa_tunnel_counts.append(tc)
            if tc > 0 and level_num is not None:
                hsa_tunnel_levels.append(level_num)
            hsa_tunnel_directions.extend(hsa.get("tunnelDirections", []))
            hsa_tunnel_queue_lengths.extend(hsa.get("tunnelQueueLengths", []))
            hsa_tunnel_queue_colors.extend(hsa.get("tunnelQueueColors", []))
            hsa_tunnel_item_keys.update(hsa.get("tunnel_item_keys", []))

        # editorMeta
        em = _extract_editor_meta(data)
        if em["present"]:
            for k in em["keys"]:
                em_key_counts[k] += 1
            for f in EM_NUMERIC_FIELDS:
                if f in em and isinstance(em[f], (int, float)):
                    em_numeric_vals[f].append(em[f])
            for f in EM_BOOL_FIELDS:
                if f in em:
                    em_bool_vals[f][em[f]] += 1
            if "zoneColors_type" in em:
                em_zoneColors_types[em["zoneColors_type"]] += 1
            if "zoneColors_len" in em:
                em_zoneColors_lens.append(em["zoneColors_len"])
            em_zoneColors_inner_lens.extend(em.get("zoneColors_inner_lens", []))
            if "colorsPerStackRatios_len" in em:
                em_colorsPerStackRatios_lens.append(em["colorsPerStackRatios_len"])
            em_colorsPerStackRatios_values.extend(
                em.get("colorsPerStackRatios_values", [])
            )

        # Colors
        all_colors = _all_colors_in_level(data)
        if level_num is not None:
            per_level_color_sets[level_num] = all_colors
        per_level_color_count.append(len(all_colors))
        global_color_counter.update(all_colors)

        # Color consistency
        declared = (data.get("editorMeta") or {}).get("colorCount")
        if isinstance(declared, int) and len(all_colors) != declared:
            color_consistency_issues.append(
                f"Level {level_num}: declared colorCount={declared}, "
                f"actual distinct colors={len(all_colors)} "
                f"(observed: {sorted(all_colors)})"
            )

    # --- Second pass: schema surprises vs level-50 baseline ----------------
    if level50_top_keys is not None:
        for level_num, data, fname in loaded_levels:
            extra = set(data.keys()) - level50_top_keys
            missing = level50_top_keys - set(data.keys())
            if extra:
                schema_surprises.append(
                    f"Level {level_num} ({fname}): extra top-level keys: {sorted(extra)}"
                )
            if missing:
                schema_surprises.append(
                    f"Level {level_num} ({fname}): missing top-level keys: {sorted(missing)}"
                )

    # CA & HSA sub-key surprises vs level 50 baseline
    ca50_keys = None
    hsa50_keys = None
    em50_keys = None
    for level_num, data, _fname in loaded_levels:
        if level_num == 50:
            ca = data.get("collectorArea") or {}
            hsa = data.get("hexStackArea") or {}
            em = data.get("editorMeta") or {}
            ca50_keys = set(ca.keys()) if isinstance(ca, dict) else None
            hsa50_keys = set(hsa.keys()) if isinstance(hsa, dict) else None
            em50_keys = set(em.keys()) if isinstance(em, dict) else None
            break

    for label, baseline, getter in (
        ("collectorArea", ca50_keys, lambda d: d.get("collectorArea")),
        ("hexStackArea", hsa50_keys, lambda d: d.get("hexStackArea")),
        ("editorMeta", em50_keys, lambda d: d.get("editorMeta")),
    ):
        if baseline is None:
            continue
        for level_num, data, _fname in loaded_levels:
            sub = getter(data)
            if not isinstance(sub, dict):
                continue
            extra = set(sub.keys()) - baseline
            missing = baseline - set(sub.keys())
            if extra:
                schema_surprises.append(
                    f"Level {level_num}: {label} has extra keys vs level-50: {sorted(extra)}"
                )
            if missing:
                schema_surprises.append(
                    f"Level {level_num}: {label} is missing keys vs level-50: {sorted(missing)}"
                )

    # --- Gap analysis -------------------------------------------------------
    all_nums = sorted(level_nums_seen)
    expected = set(range(1, 101)) if all_nums else set()
    gaps = sorted(expected - set(all_nums))
    extras = sorted(set(all_nums) - expected)

    # --- Difficulty-progression buckets ------------------------------------
    bucket_defs = [(1, 25), (26, 50), (51, 75), (76, 100)]
    bucket_stats = []
    for lo, hi in bucket_defs:
        levels_in_bucket = [
            (ln, d) for ln, d, _ in loaded_levels if ln is not None and lo <= ln <= hi
        ]
        if not levels_in_bucket:
            bucket_stats.append({"range": f"{lo}-{hi}", "count": 0})
            continue
        totals = []
        colors = []
        stacks = []
        heights_all = []
        mech_presence = {arr: 0 for arr in CA_MECHANIC_ARRAYS}
        for ln, d in levels_in_bucket:
            em = d.get("editorMeta") or {}
            if isinstance(em.get("totalBlocks"), (int, float)):
                totals.append(em["totalBlocks"])
            if isinstance(em.get("colorCount"), (int, float)):
                colors.append(em["colorCount"])
            hsa = d.get("hexStackArea") or {}
            stacks_list = hsa.get("stacks") if isinstance(hsa, dict) else None
            if isinstance(stacks_list, list):
                stacks.append(len(stacks_list))
                for s in stacks_list:
                    if isinstance(s, dict):
                        cs = s.get("colors")
                        if isinstance(cs, list):
                            heights_all.append(len(cs))
            ca = d.get("collectorArea") or {}
            if isinstance(ca, dict):
                for arr in CA_MECHANIC_ARRAYS:
                    v = ca.get(arr)
                    if isinstance(v, list) and len(v) > 0:
                        mech_presence[arr] += 1
        bucket_stats.append({
            "range": f"{lo}-{hi}",
            "count": len(levels_in_bucket),
            "totalBlocks": _stats(totals),
            "colorCount": _stats(colors),
            "stack_count": _stats(stacks),
            "stack_height": _stats(heights_all),
            "mech_presence_pct": {
                arr: round(100 * cnt / len(levels_in_bucket), 1)
                for arr, cnt in mech_presence.items()
            },
        })

    # --- Build return dict -------------------------------------------------
    return {
        "files_found": len(files),
        "files_loaded": len(loaded_levels),
        "malformed": malformed,
        "level_nums": all_nums,
        "gaps": gaps,
        "extras": extras,

        "top_level_key_counts": dict(top_level_key_counts.most_common()),
        "top_levelNumber_stats": _stats(top_levelNumber_vals),
        "top_levelVersionCode_stats": _stats(top_levelVersionCode_vals),
        "top_levelVersionCode_dist": dict(Counter(top_levelVersionCode_vals)),

        "ca_key_counts": dict(ca_key_counts.most_common()),
        "ca_gridWidth": _stats(ca_gridW),
        "ca_gridHeight": _stats(ca_gridH),

        "hsa_key_counts": dict(hsa_key_counts.most_common()),
        "hsa_gridWidth": _stats(hsa_gridW),
        "hsa_gridHeight": _stats(hsa_gridH),
        "hsa_stack_counts": _stats(hsa_stack_counts),
        "hsa_stack_heights": _stats(hsa_stack_heights),
        "hsa_stack_height_dist": _histogram(hsa_stack_heights, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "hsa_stack_item_keys": sorted(hsa_stack_item_keys),

        "mech_count_per_level_stats": {
            arr: _stats(mech_count_per_level[arr]) for arr in CA_MECHANIC_ARRAYS
        },
        "mech_count_per_level_nonzero_stats": {
            arr: _stats([c for c in mech_count_per_level[arr] if c > 0])
            for arr in CA_MECHANIC_ARRAYS
        },
        "mech_levels_with": {arr: sorted(mech_levels_with[arr]) for arr in CA_MECHANIC_ARRAYS},
        "mech_item_keys": {arr: sorted(mech_item_keys[arr]) for arr in CA_MECHANIC_ARRAYS},

        "ice_capacity_stats": _stats(ice_capacities),
        "ice_capacity_dist": dict(sorted(Counter(ice_capacities).items())),
        "pin_block_count_stats": _stats(pin_block_counts),
        "pin_block_count_dist": dict(sorted(Counter(pin_block_counts).items())),
        "pin_direction_dist": dict(sorted(Counter(pin_directions).items())),

        "ca_tunnel_directions": dict(sorted(Counter(ca_tunnel_directions).items())),
        "ca_tunnel_queue_length_stats": _stats(ca_tunnel_queue_lengths),
        "ca_tunnel_queue_length_dist": dict(sorted(Counter(ca_tunnel_queue_lengths).items())),
        "ca_tunnel_queue_color_counts": dict(Counter(ca_tunnel_queue_colors).most_common()),

        "hsa_tunnel_count_stats": _stats(hsa_tunnel_counts),
        "hsa_tunnel_levels": hsa_tunnel_levels,
        "hsa_tunnel_directions": dict(sorted(Counter(hsa_tunnel_directions).items())),
        "hsa_tunnel_queue_length_stats": _stats(hsa_tunnel_queue_lengths),
        "hsa_tunnel_item_keys": sorted(hsa_tunnel_item_keys),

        "global_colors": dict(global_color_counter.most_common()),
        "per_level_color_count_stats": _stats(per_level_color_count),
        "per_level_color_count_dist": _histogram(per_level_color_count, [2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "color_consistency_issues": color_consistency_issues,

        "em_key_counts": dict(em_key_counts.most_common()),
        "em_numeric_stats": {f: _stats(em_numeric_vals[f]) for f in EM_NUMERIC_FIELDS},
        "em_bool_dists": {f: dict(em_bool_vals[f]) for f in EM_BOOL_FIELDS},
        "em_zoneColors_types": dict(em_zoneColors_types),
        "em_zoneColors_lens_stats": _stats(em_zoneColors_lens),
        "em_zoneColors_inner_lens_stats": _stats(em_zoneColors_inner_lens),
        "em_colorsPerStackRatios_lens_stats": _stats(em_colorsPerStackRatios_lens),
        "em_colorsPerStackRatios_values_stats": _stats(em_colorsPerStackRatios_values),

        "bucket_stats": bucket_stats,
        "schema_surprises": schema_surprises,
        "level50_top_keys": sorted(level50_top_keys) if level50_top_keys else None,
        "level50_ca_keys": sorted(ca50_keys) if ca50_keys else None,
        "level50_hsa_keys": sorted(hsa50_keys) if hsa50_keys else None,
        "level50_em_keys": sorted(em50_keys) if em50_keys else None,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _render_stats_line(stats: dict) -> str:
    if not stats or stats.get("count", 0) == 0:
        return "(no data)"
    parts = [f"n={stats['count']}",
             f"min={stats['min']}", f"max={stats['max']}",
             f"mean={stats['mean']}", f"median={stats['median']}"]
    if "stdev" in stats:
        parts.append(f"stdev={stats['stdev']}")
    return ", ".join(parts)


def render_report(s: dict) -> str:
    L = []
    A = L.append

    A("# Paxie Hex Fall Level Dataset — Survey Report")
    A("")
    A(f"**Files found:** {s['files_found']}  ")
    A(f"**Files loaded successfully:** {s['files_loaded']}  ")
    A(f"**Malformed/unreadable:** {len(s['malformed'])}")
    A("")

    # ── 0. Malformed files ───────────────────────────────────────────────
    A("## 0. Malformed / Unreadable Files")
    A("")
    if s["malformed"]:
        for fname, err in s["malformed"]:
            A(f"- `{fname}`: {err}")
    else:
        A("None — all files parsed successfully.")
    A("")

    # ── 1. Level numbering coverage ──────────────────────────────────────
    A("## 1. Level Numbering Coverage")
    A("")
    nums = s["level_nums"]
    A(f"- **File-derived level numbers** (first integer in filename): {len(nums)} parseable")
    if nums:
        A(f"- **Range:** {min(nums)} – {max(nums)}")
    A(f"- **Gaps in 1–100:** {s['gaps'] if s['gaps'] else 'none'}")
    A(f"- **Numbers outside 1–100:** {s['extras'] if s['extras'] else 'none'}")
    A(f"- **In-file `levelNumber`:** {_render_stats_line(s['top_levelNumber_stats'])}")
    A(f"- **In-file `levelVersionCode`:** {_render_stats_line(s['top_levelVersionCode_stats'])}")
    A(f"- **levelVersionCode distribution:** {s['top_levelVersionCode_dist']}")
    A("")

    # ── 2. Top-level schema ──────────────────────────────────────────────
    A("## 2. Top-Level JSON Schema")
    A("")
    A(f"**Level-50 top-level keys (baseline):** `{s['level50_top_keys']}`")
    A("")
    A("| Key | # levels |")
    A("|-----|----------|")
    for k, cnt in s["top_level_key_counts"].items():
        A(f"| `{k}` | {cnt} |")
    A("")
    A(f"**Level-50 collectorArea keys:** `{s['level50_ca_keys']}`")
    A(f"**Level-50 hexStackArea keys:** `{s['level50_hsa_keys']}`")
    A(f"**Level-50 editorMeta keys:** `{s['level50_em_keys']}`")
    A("")

    A("### collectorArea sub-key frequency")
    A("")
    A("| Key | # levels |")
    A("|-----|----------|")
    for k, cnt in s["ca_key_counts"].items():
        A(f"| `{k}` | {cnt} |")
    A("")
    A("### hexStackArea sub-key frequency")
    A("")
    A("| Key | # levels |")
    A("|-----|----------|")
    for k, cnt in s["hsa_key_counts"].items():
        A(f"| `{k}` | {cnt} |")
    A("")
    A("### editorMeta field frequency")
    A("")
    A("| Field | # levels |")
    A("|-------|----------|")
    for k, cnt in s["em_key_counts"].items():
        A(f"| `{k}` | {cnt} |")
    A("")

    # ── 3. Structural distributions ──────────────────────────────────────
    A("## 3. Structural Distributions")
    A("")
    A("### collectorArea grid dimensions")
    A(f"- gridWidth: {_render_stats_line(s['ca_gridWidth'])}")
    A(f"- gridHeight: {_render_stats_line(s['ca_gridHeight'])}")
    A("")
    A("### hexStackArea grid dimensions")
    A(f"- gridWidth: {_render_stats_line(s['hsa_gridWidth'])}")
    A(f"- gridHeight: {_render_stats_line(s['hsa_gridHeight'])}")
    A("")
    A("### Stack inventory")
    A(f"- stacks per level: {_render_stats_line(s['hsa_stack_counts'])}")
    A(f"- stack item keys (across dataset): `{s['hsa_stack_item_keys']}`")
    A(f"- stack heights (all stacks pooled): {_render_stats_line(s['hsa_stack_heights'])}")
    A("")
    A("**Stack-height distribution:**")
    A("")
    A("| Height bucket | Count |")
    A("|---------------|-------|")
    for k, v in s["hsa_stack_height_dist"].items():
        A(f"| {k} | {v} |")
    A("")

    # ── 4. Mechanics frequency ───────────────────────────────────────────
    A("## 4. Mechanics Frequency")
    A("")
    A(f"Across **{s['files_loaded']}** levels, for each `collectorArea` sub-array:")
    A("")
    A("| Mechanic | # levels with ≥1 | levels-with-list (first 20) | # per level (non-zero): min/median/max/mean | item-level keys |")
    A("|----------|------------------|------------------------------|---------------------------------------------|-----------------|")
    for arr in CA_MECHANIC_ARRAYS:
        levels = s["mech_levels_with"][arr]
        nz = s["mech_count_per_level_nonzero_stats"][arr]
        keys = s["mech_item_keys"][arr]
        levels_str = str(levels[:20]) + (" …" if len(levels) > 20 else "")
        if nz.get("count", 0) > 0:
            stat = f"{nz['min']}/{nz['median']}/{nz['max']}/{nz['mean']}"
        else:
            stat = "—"
        A(f"| `{arr}` | {len(levels)} | {levels_str} | {stat} | `{keys}` |")
    A("")

    # ── 5. Mechanic details (ice / pin / tunnel) ─────────────────────────
    A("## 5. Mechanic Specifics")
    A("")
    A("### iceCollectors.iceCapacity")
    A(f"- {_render_stats_line(s['ice_capacity_stats'])}")
    A(f"- Distribution (capacity → count): {s['ice_capacity_dist']}")
    A("")
    A("### pinBlockers")
    A(f"- blockCount: {_render_stats_line(s['pin_block_count_stats'])}")
    A(f"- blockCount distribution: {s['pin_block_count_dist']}")
    A(f"- direction distribution: {s['pin_direction_dist']}")
    A("")
    A("### collectorArea.tunnels")
    A(f"- direction distribution: {s['ca_tunnel_directions']}")
    A(f"- collectorQueue length: {_render_stats_line(s['ca_tunnel_queue_length_stats'])}")
    A(f"- collectorQueue length distribution: {s['ca_tunnel_queue_length_dist']}")
    A("")
    A("**Tunnel queue color inventory (token → total occurrences across all tunnel queues):**")
    A("")
    A("| Color token | Count |")
    A("|-------------|-------|")
    for k, v in s["ca_tunnel_queue_color_counts"].items():
        A(f"| `{k}` | {v} |")
    A("")
    A("### hexStackArea.tunnels")
    A(f"- tunnel count per level: {_render_stats_line(s['hsa_tunnel_count_stats'])}")
    A(f"- levels with HSA tunnels: {s['hsa_tunnel_levels'] if s['hsa_tunnel_levels'] else 'none'}")
    A(f"- direction distribution: {s['hsa_tunnel_directions']}")
    A(f"- queue length: {_render_stats_line(s['hsa_tunnel_queue_length_stats'])}")
    A(f"- item keys (across dataset): `{s['hsa_tunnel_item_keys']}`")
    A("")

    # ── 6. Color inventory ───────────────────────────────────────────────
    A("## 6. Color-Code Inventory")
    A("")
    A("**All distinct color tokens observed across the dataset (stacks + collectors + tunnel queues + zoneColors):**")
    A("")
    A("| Color token | # levels containing it |")
    A("|-------------|------------------------|")
    for token, cnt in sorted(s["global_colors"].items(), key=lambda x: -x[1]):
        A(f"| `{token}` | {cnt} |")
    A("")
    A("### Per-level distinct color count")
    A(f"- {_render_stats_line(s['per_level_color_count_stats'])}")
    A("")
    A("| Color count bucket | # levels |")
    A("|---------------------|----------|")
    for k, v in s["per_level_color_count_dist"].items():
        A(f"| {k} | {v} |")
    A("")
    A("### editorMeta.colorCount vs. actual distinct colors")
    A("")
    if s["color_consistency_issues"]:
        for issue in s["color_consistency_issues"]:
            A(f"- {issue}")
    else:
        A("No discrepancies — every level's `editorMeta.colorCount` matches the actual distinct color count.")
    A("")

    # ── 7. editorMeta distributions ──────────────────────────────────────
    A("## 7. editorMeta Field Distributions")
    A("")
    A("### Numeric fields")
    A("")
    A("| Field | n | min | max | mean | median | stdev |")
    A("|-------|---|-----|-----|------|--------|-------|")
    for f in EM_NUMERIC_FIELDS:
        st = s["em_numeric_stats"][f]
        if st.get("count", 0) == 0:
            A(f"| `{f}` | 0 | — | — | — | — | — |")
        else:
            A(f"| `{f}` | {st['count']} | {st['min']} | {st['max']} | {st['mean']} | {st['median']} | {st.get('stdev', '—')} |")
    A("")
    A("### Boolean fields")
    A("")
    for f in EM_BOOL_FIELDS:
        A(f"- `{f}`: {s['em_bool_dists'][f]}")
    A("")
    A("### zoneColors")
    A(f"- type distribution: {s['em_zoneColors_types']}")
    A(f"- # zones per level (zoneColors_len): {_render_stats_line(s['em_zoneColors_lens_stats'])}")
    A(f"- # colors per zone (zoneColors_inner_lens): {_render_stats_line(s['em_zoneColors_inner_lens_stats'])}")
    A("")
    A("### colorsPerStackRatios")
    A(f"- # entries per level: {_render_stats_line(s['em_colorsPerStackRatios_lens_stats'])}")
    A(f"- pooled values: {_render_stats_line(s['em_colorsPerStackRatios_values_stats'])}")
    A("")

    # ── 8. Difficulty-progression buckets ────────────────────────────────
    A("## 8. Difficulty Progression (per 25-level bucket)")
    A("")
    A("Aggregates computed by bucket so the trajectory of difficulty signals can be eyeballed without revealing per-level content.")
    A("")
    A("| Range | n | totalBlocks (mean / med / max) | colorCount (mean / med / max) | stacks/level (mean) | stack height (mean) |")
    A("|-------|---|-------------------------------|------------------------------|--------------------|-----|")
    for b in s["bucket_stats"]:
        if b["count"] == 0:
            A(f"| {b['range']} | 0 | — | — | — | — |")
            continue
        tb = b["totalBlocks"]; cc = b["colorCount"]; sc = b["stack_count"]; sh = b["stack_height"]
        A(f"| {b['range']} | {b['count']} | "
          f"{tb['mean']} / {tb['median']} / {tb['max']} | "
          f"{cc['mean']} / {cc['median']} / {cc['max']} | "
          f"{sc['mean']} | {sh['mean']} |")
    A("")
    A("**Mechanic presence (% of bucket levels with ≥1 of each):**")
    A("")
    hdr_mechs = list(CA_MECHANIC_ARRAYS)
    A("| Range | " + " | ".join(f"`{m}`" for m in hdr_mechs) + " |")
    A("|-------|" + "|".join(["---"] * len(hdr_mechs)) + "|")
    for b in s["bucket_stats"]:
        if b["count"] == 0:
            A(f"| {b['range']} | " + " | ".join("—" for _ in hdr_mechs) + " |")
            continue
        cells = [f"{b['mech_presence_pct'][m]}%" for m in hdr_mechs]
        A(f"| {b['range']} | " + " | ".join(cells) + " |")
    A("")

    # ── 9. Schema surprises ──────────────────────────────────────────────
    A("## 9. Schema Surprises")
    A("")
    if s["schema_surprises"]:
        A(f"({len(s['schema_surprises'])} notes — deviations from the level-50 schema)")
        A("")
        for note in s["schema_surprises"]:
            A(f"- {note}")
    else:
        A("None — every level shares the same schema shape as level 50.")
    A("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Survey Paxie Hex Fall level dataset — aggregate report only."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("CLASSIFIED.paxie_data/level_data"),
        help="Directory containing level JSON files (default: CLASSIFIED.paxie_data/level_data)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("CLASSIFIED.paxie_data/survey_report.md"),
        help="Output path for the markdown report",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Surveying: {args.data_dir}", file=sys.stderr)
    data = survey(args.data_dir)
    report = render_report(data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Report written to: {args.out}", file=sys.stderr)
    print(f"Loaded {data['files_loaded']}/{data['files_found']} files.", file=sys.stderr)
    if data["malformed"]:
        print(f"WARNING: {len(data['malformed'])} malformed files.", file=sys.stderr)
    if data["gaps"]:
        print(f"WARNING: Gaps in level numbering: {data['gaps']}", file=sys.stderr)


if __name__ == "__main__":
    main()
