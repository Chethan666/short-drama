#!/usr/bin/env bash
# Restores data/ from the newest story database any earlier run saved.
#
# Picks by the artifact's own upload time across the recent runs, whatever status
# GitHub shows for the run: run statuses can lag minutes behind (a run that has
# already saved its database may still be listed as queued or in progress), and
# trusting them once handed a run an older database and lost an approved episode.
# The repository-wide artifact list can lag too, so artifacts are read run by run.
# Needs GH_TOKEN, REPO and GITHUB_RUN_ID.
set -euo pipefail

runs=$(gh api "repos/$REPO/actions/workflows/short-drama.yml/runs?per_page=12" --jq \
  "[.workflow_runs[] | select(.id != $GITHUB_RUN_ID)] | .[].id")

best_id="" best_at="" best_run=""
for attempt in 1 2 3; do
  for run in $runs; do
    line=$(gh api "repos/$REPO/actions/runs/$run/artifacts" --jq \
      '[.artifacts[] | select(.expired == false and (.name == "story-db" or (.name | startswith("stories-"))))]
       | sort_by(.created_at) | last | select(. != null) | "\(.created_at) \(.id)"' || true)
    [ -z "$line" ] && continue
    at=${line% *}; id=${line#* }
    if [[ -z "$best_at" || "$at" > "$best_at" ]]; then
      best_at=$at; best_id=$id; best_run=$run
    fi
  done
  [ -n "$best_id" ] && break
  sleep 10  # nothing found yet: a just-finished upload may still be settling
done

if [ -z "$best_id" ]; then
  echo "No saved story database yet; starting a new one"
  exit 0
fi
mkdir -p data
gh api "repos/$REPO/actions/artifacts/$best_id/zip" > "$RUNNER_TEMP/story-db.zip"
unzip -o "$RUNNER_TEMP/story-db.zip" -d data
echo "Restored the story database saved by run $best_run at $best_at (artifact $best_id)"
