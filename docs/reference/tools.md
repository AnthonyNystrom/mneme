# Tools

Operational scripts shipped with `mneme`. Both have Python APIs and `python -m` CLIs.

## Calibration

`mneme.tools.calibrate` finds the right `similarity_threshold` for your embedder + corpus.

::: mneme.tools.calibrate.find_threshold

::: mneme.tools.calibrate.precision_recall_curve

::: mneme.tools.calibrate.CalibrationResult

### CLI

```bash
python -m mneme.tools.calibrate --help
```

## Migration

`mneme.tools.migrate` re-embeds an existing cache through a new embedder when you switch model or dimension.

::: mneme.tools.migrate.reembed

::: mneme.tools.migrate.areembed
