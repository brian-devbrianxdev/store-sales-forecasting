"""Command-line orchestrator for the store-sales pipeline.

Replaces the notebook's environment-variable ``run()`` calls with explicit
subcommands. Installed as the ``store-sales`` console script.

Examples:
    store-sales train lgbm-v8 --suffix reg
    store-sales train darts-family --variant deeper
    store-sales train tsmixer --epochs 30
    store-sales train chronos2-cov --suffix _promo --oil --holiday   # needs .venv_chronos2
    store-sales blend build
    store-sales blend verify
    store-sales run-all                       # every CPU/GPU leg, then the blend

Each ``train`` leg forwards its remaining arguments to that leg's own parser, so
all per-leg flags remain available.
"""
from __future__ import annotations

import argparse
import sys

from .config import get_config


def _cmd_train(leg: str, rest: list[str]) -> None:
    """Dispatch a ``train <leg>`` invocation to the matching leg ``main``."""
    cfg = get_config()
    if leg == "lgbm-v8":
        from .models import lgbm_regularized
        lgbm_regularized.main(rest)
    elif leg == "catboost":
        from .models import catboost_family
        catboost_family.main(rest)
    elif leg == "darts-family":
        from .models import darts_family
        darts_family.main(rest)
    elif leg == "tsmixer":
        # Default the output name to the tuned filename the ensemble expects.
        from .models import neural_ts
        argv = ["--model", "tsmixer"]
        if not any(a == "--out-name" for a in rest):
            argv += ["--out-name", cfg.neural.tuned_out_name]
        neural_ts.main(argv + rest)
    elif leg == "neural":
        from .models import neural_ts
        neural_ts.main(rest)
    elif leg == "chronos2":
        from .models import chronos2
        chronos2.main(rest)
    elif leg == "chronos2-cov":
        from .models import chronos2_cov
        chronos2_cov.main(rest)
    else:  # pragma: no cover - argparse choices guard this
        raise SystemExit(f"unknown leg: {leg}")


def _cmd_blend(action: str) -> None:
    """Dispatch a ``blend <action>`` invocation."""
    from .ensemble import build
    if action == "build":
        build.run_build()
    elif action == "verify":
        ok = build.verify()
        sys.exit(0 if ok else 1)
    elif action == "oilhol-swap":
        from .ensemble import alternates
        alternates.run_oilhol_swap()
    elif action == "positive-hedge":
        from .ensemble import alternates
        alternates.run_positive_hedge()


def _cmd_run_all() -> None:
    """Train every in-process leg (darts family + v8 + tsmixer) then build.

    The Chronos-2 covariate leg is NOT run here — it requires the separate
    ``.venv_chronos2`` interpreter. Its committed CSV is used by the blend.
    """
    cfg = get_config()
    from .models import darts_family, lgbm_regularized, neural_ts
    from .ensemble import build

    for variant in cfg.darts_family.variants:
        print(f"\n===== darts-family: {variant} =====", flush=True)
        darts_family.run(darts_family.DartsSettings.from_variant(variant))

    print("\n===== lgbm-v8 =====", flush=True)
    lgbm_regularized.main(["--suffix", "reg"])

    print("\n===== tsmixer =====", flush=True)
    neural_ts.run("tsmixer", cfg.neural.default_epochs, gpu=True,
                  out_name=cfg.neural.tuned_out_name)

    print("\n===== blend =====", flush=True)
    build.run_build()


def main(argv: list[str] | None = None) -> None:
    """Parse the top-level command and dispatch.

    Unknown trailing arguments after ``train <leg>`` are forwarded verbatim to
    the leg's own argument parser.
    """
    cfg = get_config()
    parser = argparse.ArgumentParser(prog="store-sales", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a single leg")
    p_train.add_argument("leg", choices=[
        "lgbm-v8", "catboost", "darts-family", "tsmixer", "neural",
        "chronos2", "chronos2-cov",
    ])

    p_blend = sub.add_parser("blend", help="build/verify the ensemble blend or alternates")
    p_blend.add_argument("action",
                         choices=["build", "verify", "oilhol-swap", "positive-hedge"])

    sub.add_parser("run-all", help="train every in-process leg then build the blend")

    # Split argv so leg-specific flags are not consumed by the top-level parser.
    args, rest = parser.parse_known_args(argv)

    if args.command == "train":
        _cmd_train(args.leg, rest)
    elif args.command == "blend":
        _cmd_blend(args.action)
    elif args.command == "run-all":
        _cmd_run_all()


if __name__ == "__main__":
    main()
