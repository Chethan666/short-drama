# Short drama

Vertical AI short-drama series (60–120-second episodes with a hook, a reversal and a
cliffhanger), moved out of the Hollow Lines channel repository so the two can grow apart.

| # | Pipeline | What it makes | Code | Docs |
|---|---|---|---|---|
| 3 | Drama writers' room | series bibles and dialogue episodes | `drama_factory/` | [DRAMA_FACTORY.md](DRAMA_FACTORY.md) |
| 4 | Drama production | TV-style vertical episodes from those scripts | `drama_production/` | [DRAMA_PRODUCTION.md](DRAMA_PRODUCTION.md) |

`story_factory/` is the shared engine both use: model access (Amazon Bedrock and direct
APIs), judging, the database and the cost log. It is a copy of the one in the channel
repository ([STORY_FACTORY.md](STORY_FACTORY.md)); the two copies may now diverge.

Everything runs from the **Short drama** workflow in the Actions tab; the modes are listed at
the top of `.github/workflows/short-drama.yml`.

## Where things stand

- **fal.ai path** (`drama-video`): about $2.60 per finished minute (Seedream v4 pictures,
  Seedance 1.0 lite clips).
- **Open-model path** (`drama-vast`, a rented Vast.ai GPU with no download charges): the ready
  script `drama_production/scripts/the-last-call.json` ("The Last Call", 18 shots, 57 s) has
  been made with Chatterbox voices, Qwen-Image-Edit-2511 pictures, Wan 2.2 I2V motion on every
  shot and MMAudio room tone (about $6 of GPU time). Lip sync (Wan2.2-S2V) works but takes about
  23 minutes a shot on an RTX PRO 6000; one of the nine speaking shots is lip-synced so far.
  Finishing the other eight is about 3.2 GPU hours (~$5.70), reusing everything else with
  `reuse_run`.

## Secrets

GitHub Actions secrets (Settings → Secrets and variables → Actions): `AWS_BEARER_TOKEN_BEDROCK`
(writing), `FAL_KEY` and optionally `BOSON_API_KEY` (fal.ai video), `VAST_API_KEY` (open
models on Vast.ai).

Episodes written in the channel repository's database did not move with the code; the ready
script needs no database.
