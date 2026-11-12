"""Command line for pipeline 4, the drama video pipeline.

    python -m drama_production check              # what each configured model expects; Boson voices
    python -m drama_production produce            # the next approved episode's video
    python -m drama_production produce --episode 7
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from story_factory import db
from story_factory.__main__ import apply_overrides
from story_factory.config import load_config

from .media import BosonVoice, Fal, MediaError

log = logging.getLogger("drama_production")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drama_production", description="Pipeline 4: drama videos")
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="list each configured fal.ai model's inputs and the Boson voices (no spend)")
    p = sub.add_parser("produce", help="make the video for an approved episode")
    p.add_argument("--episode", type=int, help="a chosen episode id (default: the next approved one)")
    p.add_argument("--out", default="out")
    p.add_argument("--voice", choices=("auto", "kokoro", "boson"), default="auto")
    v = sub.add_parser("vast", help="make an episode with open models on a rented Vast.ai GPU")
    v.add_argument("--episode", type=int, help="an episode id from the database")
    v.add_argument("--script", default=str(Path(__file__).with_name("scripts") / "the-last-call.json"),
                   help="a ready episode script (bible + episode), used when no --episode is given")
    v.add_argument("--out", default="out")
    v.add_argument("--max-usd", type=float, dest="max_usd")
    v.add_argument("--reuse", help="a folder of pictures an earlier run made (ref_*.png, shot_*.png) to skip redrawing")
    sub.add_parser("vast-check", help="the Vast.ai balance and the best GPU offers, with their traffic charges (no spend)")
    sub.add_parser("vast-cleanup", help="destroy any Vast.ai machine this run rented")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = apply_overrides(load_config(args.config), args.set)
    d = cfg["drama_video"]

    if args.command == "check":
        failed = 0
        print("FAL_KEY:", "set" if os.environ.get("FAL_KEY") else "MISSING (add it as a GitHub Actions secret)")
        print("BOSON_API_KEY:", "set" if os.environ.get("BOSON_API_KEY") else "MISSING (voices fall back to Kokoro)")
        fal = Fal(key=os.environ.get("FAL_KEY") or "not-needed-to-read-schemas")
        for role, endpoint in d["models"].items():
            try:
                info = fal.schema(endpoint)
                print(f"{role:<9} {endpoint}\n          inputs: {', '.join(info['fields'])}\n"
                      f"          required: {', '.join(info['required'])}")
                for field in ("duration", "resolution", "aspect_ratio", "num_frames"):
                    if field in info.get("options", {}):
                        print(f"          {field}: {info['options'][field]}")
                unknown = set(d["model_args"].get(role, {})) - set(info["fields"]) if info["fields"] else set()
                if unknown:
                    print(f"          WARNING: configured arguments the model doesn't list: {sorted(unknown)}")
            except MediaError as exc:
                failed += 1
                print(f"{role:<9} {endpoint}  NOT FOUND: {exc}")
        try:
            boson = BosonVoice(d["boson_model"])
        except MediaError as exc:
            print(f"Boson: {exc}")
        else:
            for label, call in (("models", boson.models), ("voices", boson.voices)):
                try:
                    print(f"Boson {label}:", json.dumps(call(), indent=2)[:4000])
                except MediaError as exc:
                    print(f"Boson {label}: {exc}")
            print(f"Boson speech with {d['boson_model']}:")
            for name, outcome in boson.probe(list(d.get("boson_probe_voices", []))).items():
                print(f"  {name:<18} {outcome}")
        return 1 if failed else 0

    if args.command == "vast":
        return vast(args, cfg)
    if args.command in ("vast-check", "vast-cleanup"):
        return vast_admin(args.command)

    from .produce import Producer  # the video libraries are only needed here

    conn = db.connect(cfg["database"])
    row = (conn.execute("SELECT * FROM episodes WHERE id = ?", (args.episode,)).fetchone() if args.episode
           else conn.execute("SELECT * FROM episodes WHERE status = 'approved' ORDER BY series_id, number LIMIT 1").fetchone())
    if row is None:
        print("No approved episode waiting", file=sys.stderr)
        return 1
    from .voices import KokoroVoice

    voices = KokoroVoice("am_michael", cfg["video"].get("lang", "a"), 1.0)  # every character's fixed voice
    boson = None
    if args.voice == "boson" or (args.voice == "auto" and os.environ.get("BOSON_API_KEY")):
        boson = BosonVoice(d["boson_model"])  # expressive delivery, cloned from each fixed voice
    assets = Path(cfg["database"]).parent / "series" / str(row["series_id"])
    try:
        meta = Producer(conn, cfg, assets, Fal(), voices, boson=boson).produce(row["id"], Path(args.out))
    except Exception as exc:  # record and report; the episode stays approved to retry
        log.exception("Episode production failed")
        conn.execute("UPDATE episodes SET production = ?, updated_at = ? WHERE id = ?",
                     (json.dumps({"error": str(exc)[:500], "run": os.environ.get("GITHUB_RUN_ID")}), db.now(), row["id"]))
        print(f"[failed] episode #{row['id']}: {exc}", file=sys.stderr)
        return 2
    meta["run"] = os.environ.get("GITHUB_RUN_ID")
    status = {"status": "produced"} if row["status"] == "approved" else {}
    conn.execute("UPDATE episodes SET production = ?, updated_at = ?" + (", status = 'produced'" if status else "")
                 + " WHERE id = ?", (json.dumps(meta), db.now(), row["id"]))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


def _label() -> str:
    return f"short-drama-{os.environ.get('GITHUB_RUN_ID', 'local')}"


def vast_admin(command: str) -> int:
    from .vast import QUERIES, Vast, download_usd, expected_cost, upload_usd

    vast = Vast(label=_label())
    if command == "vast-cleanup":
        print(f"Destroyed {vast.cleanup_label()} leftover machine(s) labelled {vast.label}")
        return 0
    print("Balance:", vast.balance())
    try:
        user = vast.cli("show", "user")
        print("Account:", {k: user.get(k) for k in ("credit", "balance", "balance_threshold", "has_billing",
                                                    "billing_creditonly", "can_pay") if isinstance(user, dict)})
        mine = vast.cli("show", "instances") or []
        print(f"Your machines ({len(mine)}):")
        for inst in mine:
            print(f"  id {inst.get('id')} {inst.get('gpu_name')} status {inst.get('actual_status')} "
                  f"label {inst.get('label')} ${float(inst.get('dph_total') or 0):.2f}/h")
    except Exception as exc:  # noqa: BLE001 - a report
        print("Account details unavailable:", exc)
    offers = vast.offers()
    print(f"{len(offers)} suitable offers; the cheapest:")
    for o in offers[:8]:
        print(f"  id {o.get('id')}  {o.get('gpu_name')}  ${float(o.get('dph_total') or 0):.2f}/h  "
              f"gpu_ram {o.get('gpu_ram')}  cpu_ram {o.get('cpu_ram')}  disk {o.get('disk_space')}  "
              f"down {o.get('inet_down')}  reliability {o.get('reliability2', o.get('reliability'))}  "
              f"cuda {o.get('cuda_max_good')}  traffic ~${download_usd(o) + upload_usd(o):.2f}  "
              f"typical run ~${expected_cost(o):.2f}")
    if offers:
        print("Offer fields:", ", ".join(sorted(offers[0])))
    # Why offers were left out: the cheapest 80 GB machines before the traffic rule, with their charges.
    for label, query in (("our filters", QUERIES[-1]),
                         ("any verified 80 GB", "num_gpus=1 gpu_ram>=79 verified=true rentable=true")):
        try:
            raw = vast.cli("search", "offers", query, "-o", "dph_total")
        except Exception as exc:  # noqa: BLE001 - a report
            print(f"{label}: search failed: {exc}")
            continue
        raw = raw if isinstance(raw, list) else []
        free = [o for o in raw if download_usd(o) == 0 and upload_usd(o) == 0]
        print(f"{label}: {len(raw)} offers, {len(free)} with no traffic charges; the cheapest:")
        for o in raw[:12]:
            print(f"  id {o.get('id')}  {o.get('gpu_name')}  ${float(o.get('dph_total') or 0):.2f}/h  "
                  f"disk {o.get('disk_space')}  down {o.get('inet_down')}  arch {o.get('cpu_arch')}  "
                  f"down $/TB {o.get('internet_down_cost_per_tb')} (per GB {o.get('inet_down_cost')})  "
                  f"up $/TB {o.get('internet_up_cost_per_tb')} (per GB {o.get('inet_up_cost')})  "
                  f"-> traffic for this job ~${download_usd(o) + upload_usd(o):.2f}")
    return 0


def vast(args: argparse.Namespace, cfg: dict) -> int:
    """Open models on a rented GPU: Chatterbox, Qwen-Image-Edit-2511, Wan2.2-S2V, Wan 2.2 I2V, MMAudio."""
    from .vast import Vast

    from . import vastdrama
    from .voices import KokoroVoice

    conn = db.connect(cfg["database"])
    if args.episode:
        row = conn.execute("SELECT * FROM episodes WHERE id = ?", (args.episode,)).fetchone()
        if row is None:
            print(f"No episode #{args.episode}", file=sys.stderr)
            return 1
        series = conn.execute("SELECT * FROM series WHERE id = ?", (row["series_id"],)).fetchone()
        bible, episode = json.loads(series["bible"]), json.loads(row["script"])
        bible.setdefault("title", series["title"])
        name = f"series-{series['id']}-ep-{row['number']:02d}"
    else:
        script = json.loads(Path(args.script).read_text())
        bible, episode = script["bible"], script["episode"]
        name = Path(args.script).stem
    print(f"Open-model drama on Vast.ai: {bible.get('title')} - {episode.get('title')} "
          f"({len(episode.get('shots', []))} shots)", flush=True)
    vcfg = dict(cfg.get("drama_vast", {}))
    if args.max_usd is not None:
        vcfg["max_usd"] = args.max_usd
    kokoro = KokoroVoice("af_heart", cfg["video"].get("lang", "a"), 1.0)
    summary = vastdrama.produce(bible, episode, Path(args.out) / f"drama-vast-{name}", kokoro, vcfg,
                                Vast(label=_label()), Path(__file__).with_name("vast_drama_worker.py"), full_cfg=cfg,
                                reuse=Path(args.reuse) if args.reuse else None)
    if summary.get("usd"):
        db.record_cost(conn, pipeline="drama", task="vast_gpu", model=str(summary["gpu"].get("gpu")), input_units=0,
                       output_units=summary["moving_shots"], usd=float(summary["usd"]))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
