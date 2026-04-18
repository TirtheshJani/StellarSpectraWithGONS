#!/usr/bin/env bash
# Citation-hygiene grep gate for the interpretability module.
#
# Fails (exit 1) if a stale "Liu et al 2019" reference sneaks into source,
# scripts, tests, or README. The correct attribution for the underpinning
# work is Li, Lin & Qiu (2019). See plan-stellar-wild-manatee.md §16.
set -u

patterns=(
  'Liu et al\.?[[:space:]]*2019'
  'Liu 2019'
)

fail=0
for p in "${patterns[@]}"; do
  hits=$(grep -RIn --include='*.py' --include='*.md' --include='*.rst' \
            --include='*.ipynb' --include='*.txt' --include='*.toml' \
            -E "$p" src scripts tests README* docs 2>/dev/null \
         | grep -v 'scripts/check_citations.sh' || true)
  if [ -n "$hits" ]; then
    echo "ERROR: stale citation pattern '$p' found:" >&2
    echo "$hits" >&2
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "citation grep gate: PASS"
fi
exit "$fail"
