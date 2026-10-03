# Scripts

The supported artifact entry points are:

- `patch_fuzzers.sh SyzPilot` clones pinned upstream Syzkaller and applies the
  reviewed SyzPilot-Fuzzer patch. Set `SYZPILOT_FUZZER_DIR` only when the
  default `/root/fuzzers/SyzPilot-fuzzer` is unsuitable.
- `verify_setup.sh brain|fuzzer` checks the role-specific runtime contract
  inside a container created from the unified SyzPilot image.
- `prepare_functional_config.py` converts freshly resolved mini-benchmark
  waypoint PCs into a one-guest manager configuration for a local or
  distributed Brain/Fuzzer deployment (`--case 21`, `25`, or `36`).

The remaining scripts are focused dataset preparation, parsing, and analysis
utilities rather than installation entry points.

## Syzbot Reproducer Statistics

`syzbot_repro_stats_standalone.py` fetches the live upstream Open, Fixed,
and Invalid bug lists, prints the C+syz, syz-only, and missing-reproducer
distribution, and writes the same table to a timestamped CSV file.

```bash
pip install requests beautifulsoup4 lxml rich
python scripts/syzbot_repro_stats_standalone.py
```
