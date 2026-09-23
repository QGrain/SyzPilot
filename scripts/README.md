# Scripts

Utility scripts for dataset preparation, parsing, and analysis.

## Syzbot Reproducer Statistics

`syzbot_repro_stats_standalone.py` fetches the live upstream Open, Fixed,
and Invalid bug lists, prints the C+syz, syz-only, and missing-reproducer
distribution, and writes the same table to a timestamped CSV file.

```bash
pip install requests beautifulsoup4 lxml rich
python scripts/syzbot_repro_stats_standalone.py
```
