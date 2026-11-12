"""Pipeline 4 on open models: the job, the talking-shot rule, free-traffic hosts, and a whole episode with a fake GPU."""

import json
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

try:
    import numpy as np
    import soundfile
    from PIL import Image
    HAVE = shutil.which("ffmpeg") is not None
except ImportError:  # pragma: no cover
    HAVE = False

SCRIPT = Path(__file__).resolve().parent.parent / "drama_production" / "scripts" / "the-last-call.json"


class FakeKokoro:
    sample_rate = 24_000
    voice = "af_heart"

    def say(self, text, speed):
        return (0.1 * np.sin(np.linspace(0, 400, int(0.32 * len(text.split()) * 24_000)))).astype(np.float32)


class ScriptTests(unittest.TestCase):
    def test_the_ready_script_runs_about_a_minute_and_every_name_is_known(self):
        script = json.loads(SCRIPT.read_text())
        cast = {c["name"] for c in script["bible"]["cast"]}
        places = {loc["name"] for loc in script["bible"]["locations"]}
        shots = script["episode"]["shots"]
        words = sum(len(str(s.get("line", "")).split()) for s in shots)
        silent = sum(1 for s in shots if not s.get("line"))
        estimate = words / 2.3 + 0.6 * (len(shots) - silent) + 2.8 * silent
        self.assertTrue(50 <= estimate <= 70, estimate)
        for s in shots:
            self.assertIn(s["location"], places)
            self.assertTrue(set(s["characters"]) <= cast, s)
            if s.get("speaker"):
                self.assertIn(s["speaker"], cast)


class JobTests(unittest.TestCase):
    def test_on_screen_speakers_are_lip_synced_and_everything_else_moves(self):
        from drama_production.vastdrama import build_job, talks

        script = json.loads(SCRIPT.read_text())
        job = build_job(script["bible"], script["episode"]["shots"], {})
        modes = {s["index"]: s["mode"] for s in job["shots"]}
        shots = script["episode"]["shots"]
        self.assertEqual(modes[2], "talk")  # Maya on screen, speaking
        self.assertEqual(modes[3], "move")  # the caller is only heard
        self.assertEqual(modes[6], "move")  # a thought: the lips don't move
        self.assertTrue(all((modes[i] == "talk") == talks(s) for i, s in enumerate(shots)))
        # The double is drawn from the original's portrait, so the original comes first.
        names = [r["name"] for r in job["references"]]
        self.assertLess(names.index("Maya"), names.index("The Other Maya"))
        self.assertEqual(job["references"][names.index("The Other Maya")]["based_on"], "Maya")
        # The double speaks with Maya's voice.
        line = next(x for x in job["lines"] if x["index"] == 3)
        self.assertEqual(line["reference"], "voice_maya.wav")
        self.assertIn("{ref:Maya}", job["shots"][2]["picture_prompt"])
        self.assertLessEqual(max(len(s["refs"]) for s in job["shots"]), 3)

    def test_the_remote_script_is_valid_bash_and_names_every_model(self):
        from drama_production.vastdrama import build_job, remote_script

        script = json.loads(SCRIPT.read_text())
        text = remote_script(build_job(script["bible"], script["episode"]["shots"], {}))
        for model in ("Qwen/Qwen-Image-Edit-2511", "Wan-AI/Wan2.2-S2V-14B", "Wan-AI/Wan2.2-I2V-A14B-Diffusers"):
            self.assertIn(model, text)
        self.assertNotIn("@", text.replace("@main", ""))
        check = subprocess.run(["bash", "-n"], input=text, capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)

    def test_the_worker_compiles_on_its_own(self):
        worker = SCRIPT.parent.parent / "vast_drama_worker.py"
        compile(worker.read_text(), str(worker), "exec")
        self.assertNotIn("from drama_production", worker.read_text())


class TrafficTests(unittest.TestCase):
    def test_only_hosts_with_free_downloads_and_pennies_of_upload_are_offered(self):
        from drama_production.vast import Vast

        offers = [
            {"id": 1, "dph_total": 1.0, "inet_down_cost": 0.04, "inet_up_cost": 0.0, "inet_down": 900},
            {"id": 2, "dph_total": 1.4, "inet_down_cost": 0.0, "inet_up_cost": 0.05, "inet_down": 900},
            {"id": 5, "dph_total": 1.74, "inet_down_cost": 0.0, "inet_up_cost": 0.0133, "inet_down": 58000},
            {"id": 3, "dph_total": 1.3, "internet_down_cost_per_tb": 0, "internet_up_cost_per_tb": 0, "inet_down": 900},
            {"id": 4, "dph_total": 1.2, "inet_down_cost": 0.0, "inet_up_cost": 0.0, "inet_down": 900},
        ]

        class Fake(Vast):
            def __init__(self):
                pass

            def cli(self, *args, **kwargs):
                return offers

        self.assertEqual(sorted(o["id"] for o in Fake().offers()), [3, 4, 5])

    def test_a_slow_line_costs_more_rented_time(self):
        from drama_production.vast import expected_cost

        fast = {"dph_total": 1.3, "inet_down": 8000}
        slow = {"dph_total": 1.2, "inet_down": 500}
        self.assertLess(expected_cost(fast), expected_cost(slow))


@unittest.skipUnless(HAVE, "needs numpy, soundfile, Pillow and ffmpeg")
class EpisodeTests(unittest.TestCase):
    def test_a_whole_episode_with_a_fake_gpu_and_gaps_it_must_fill(self):
        from drama_production import vastdrama

        script = json.loads(SCRIPT.read_text())
        shots = script["episode"]["shots"]
        seen = {}

        def fake_run_job(vast, job, worker, local_out, max_usd, disk=300, max_hours=3.4, script=None):
            seen.update(job=job, script=script, max_usd=max_usd, disk=disk)
            out = local_out / "out"
            out.mkdir(parents=True)
            for i in range(len(shots)):
                if i == 4:
                    continue  # a missing picture borrows its neighbour
                Image.new("RGB", (232, 416), (20 * (i % 10), 40, 60)).save(out / f"shot_{i:03d}.png")
            for line in job["lines"][:-1]:  # the last line falls back to Kokoro
                soundfile.write(out / f"line_{line['index']:03d}.wav", np.zeros(int(2.4 * 24_000), np.float32), 24_000)
            for i in (0, 2, 16):
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                                "testsrc=size=180x320:rate=16:duration=2", str(out / f"clip_{i:03d}.mp4")], check=True)
            soundfile.write(out / "room_dispatch-center.flac", np.zeros(24_000, np.float32) + 0.01, 24_000)
            return {"gpu": "fake", "usd": 4.2, "hours": 3.1, "log": "ok"}

        with tempfile.TemporaryDirectory() as tmp:
            summary = vastdrama.produce(script["bible"], script["episode"], Path(tmp) / "ep", FakeKokoro(),
                                        {"max_usd": 9}, None, Path("w.py"), run_job=fake_run_job)
            self.assertTrue((Path(tmp) / "ep" / "episode.mp4").stat().st_size > 10_000)
            self.assertTrue((Path(tmp) / "ep" / "thumbnail.png").exists())
            self.assertFalse((Path(tmp) / "ep" / "work").exists())
        self.assertEqual(seen["max_usd"], 9)
        self.assertEqual(seen["disk"], 450)
        self.assertIn("vast_worker.py job.json talk", seen["script"])
        self.assertEqual(summary["shots"], len(shots))
        self.assertEqual(summary["moving_shots"], 3)
        self.assertEqual(summary["voices"]["kokoro"], 1)
        self.assertEqual(summary["usd"], 4.2)
        self.assertGreater(summary["seconds"], 40)


if __name__ == "__main__":
    unittest.main()


class StartTests(unittest.TestCase):
    def test_a_start_command_that_holds_the_connection_still_counts_once_the_log_exists(self):
        from drama_production import vast as vastmod

        calls = []

        class Fake(vastmod.Vast):
            def __init__(self):
                pass

            def ssh(self, machine, command, timeout=600, check=True):
                calls.append(command)
                if "setsid" in command:
                    raise subprocess.TimeoutExpired(command, timeout)
                return subprocess.CompletedProcess(command, 0, "STARTED\n", "")

        Fake().start(vastmod.Machine(1, 1.0, "x"), "/workspace/job", "run.sh", "run.log")
        self.assertEqual(len(calls), 2)
        self.assertIn("test -e /workspace/job/run.log", calls[1])

    def test_no_log_means_the_job_never_started(self):
        from drama_production import vast as vastmod

        class Fake(vastmod.Vast):
            def __init__(self):
                pass

            def ssh(self, machine, command, timeout=600, check=True):
                return subprocess.CompletedProcess(command, 0, "", "")

        with unittest.mock.patch.object(vastmod.time, "sleep"):
            with self.assertRaises(vastmod.VastError):
                Fake().start(vastmod.Machine(1, 1.0, "x"), "/w", "run.sh", "run.log")


class TalkFallbackTests(unittest.TestCase):
    def _run_talk(self, succeed):
        import importlib.util
        worker = SCRIPT.parent.parent / "vast_drama_worker.py"
        spec = importlib.util.spec_from_file_location("vast_drama_worker_t", worker)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            out, repo = Path(tmp) / "out", Path(tmp) / "repo"
            out.mkdir(), repo.mkdir()
            (out / "shot_002.png").write_bytes(b"png")
            (out / "line_002.wav").write_bytes(b"wav")
            (out / "clip_002.mp4").write_bytes(b"earlier moving clip")

            def fake_run(args, cwd=None, capture_output=True, text=True):
                if succeed:
                    Path(args[args.index("--save_file") + 1]).write_bytes(b"lip-synced clip")
                    return subprocess.CompletedProcess(args, 0, "", "")
                return subprocess.CompletedProcess(args, 1, "", "ModuleNotFoundError: easydict")

            job = {"wan_repo": str(repo), "talk_dir": "/m", "shots": [
                {"index": 2, "mode": "talk", "motion_prompt": "talks"}]}
            with unittest.mock.patch.object(mod.subprocess, "run", fake_run):
                mod.talk(job, out)
            self.assertEqual(sorted(p.name for p in out.glob("clip_002*.mp4")), ["clip_002.mp4"])
            return (out / "clip_002.mp4").read_bytes(), (out / "clip_002.talk").exists()

    def test_a_failed_lip_sync_keeps_the_earlier_moving_clip(self):
        self.assertEqual(self._run_talk(False), (b"earlier moving clip", False))

    def test_a_lip_synced_clip_replaces_it_and_is_marked(self):
        self.assertEqual(self._run_talk(True), (b"lip-synced clip", True))
