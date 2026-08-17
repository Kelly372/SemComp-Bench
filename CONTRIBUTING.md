# Contributing

Thank you for helping improve the SemComp-Data construction pipeline.

## Before opening a change

- Open an issue for substantial behavioral or schema changes so the intended
  result can be agreed on first.
- Do not commit API keys, provider responses containing private data, source
  videos, generated frames/clips, model checkpoints, or dataset artifacts.
- Preserve stage input/output compatibility unless the change explicitly
  documents a migration.
- Keep third-party code and its license notices clearly separated from
  Apache-licensed SemComp code.

## Development checks

Use Python 3.10 or newer. Before submitting a change, run:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

For changes to a processing stage, also run a small representative sample with
`--limit` and report the command, model identifier, and relevant output counts.
Do not attach private media or credentials to an issue or pull request.

## Pull requests

Describe:

1. what changed and why;
2. whether Parquet columns, prompts, or default paths changed;
3. the checks performed;
4. any data, model, or third-party license implications.

By submitting a contribution, you agree that your original contribution may
be distributed under the Apache License 2.0. Do not submit material that you
do not have permission to redistribute.
