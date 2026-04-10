#!/usr/bin/env python
"""Unified CLI entry point for the APM corridor evaluation pipeline.

After ``pip install -e .``, run::

    urbansim-apm search --help
    urbansim-apm feedback --scenario current_zoning
    urbansim-apm evaluate --scenario current_zoning
    urbansim-apm validate --auto-calibrate
    urbansim-apm decide --scenario current_zoning
    urbansim-apm sensitivity --scenario current_zoning --n-points 20
    urbansim-apm uncertainty --scenario current_zoning

Each subcommand delegates to the existing script's ``main()`` function,
preserving full backward compatibility with direct script invocation.
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)

__version__ = "1.0.0-apm"


def _cmd_search(argv: list[str]) -> int:
    """Run corridor search (NSGA-II + dense core generation)."""
    sys.argv = ["optimized_corridor_search"] + argv
    from scripts.optimized_corridor_search import main
    main()
    return 0


def _cmd_feedback(argv: list[str]) -> int:
    """Run feedback loop (25-year land use / transport simulation)."""
    sys.argv = ["run_feedback_loop"] + argv
    from scripts.run_feedback_loop import main
    main()
    return 0


def _cmd_evaluate(argv: list[str]) -> int:
    """Run integrated static + dynamic financial evaluation."""
    sys.argv = ["apm_corridor_evaluation_integrated"] + argv
    from scripts.apm_corridor_evaluation_integrated import main
    main()
    return 0


def _cmd_validate(argv: list[str]) -> int:
    """Run behavioral validation gate (TCRP/NHTS benchmarks)."""
    sys.argv = ["validate_against_benchmarks"] + argv
    from scripts.validate_against_benchmarks import main
    return main()


def _cmd_decide(argv: list[str]) -> int:
    """Generate decision package (maps, tables, report)."""
    sys.argv = ["generate_decision_package"] + argv
    from scripts.generate_decision_package import main
    return main()


def _cmd_sensitivity(argv: list[str]) -> int:
    """Run behavioral LHS sensitivity (GP surrogate fitting)."""
    sys.argv = ["run_behavioral_sensitivity"] + argv
    from scripts.run_behavioral_sensitivity import main
    return main()


def _cmd_uncertainty(argv: list[str]) -> int:
    """Run Monte Carlo uncertainty analysis (copula sampling)."""
    sys.argv = ["run_uncertainty_sensitivity"] + argv
    from scripts.run_uncertainty_sensitivity import main
    main()
    return 0


def _cmd_version(argv: list[str]) -> int:
    """Print version and environment info."""
    from src.reproducibility import get_git_commit_short, environment_fingerprint
    fp = environment_fingerprint()
    logger.info(f"urbansim-apm {__version__}")
    logger.info(f"  git: {fp.get('git_commit', 'unknown')}"
                f"{' (dirty)' if fp.get('git_dirty') else ''}")
    logger.info(f"  python: {fp.get('python_version', 'unknown').split()[0]}")
    logger.info(f"  platform: {fp.get('platform', 'unknown')}")
    key_pkgs = fp.get("key_packages", {})
    if key_pkgs:
        logger.info(f"  packages: {', '.join(f'{k}={v}' for k, v in sorted(key_pkgs.items()))}")
    return 0


_COMMANDS = {
    "search": (_cmd_search, "Run corridor search (NSGA-II optimization)"),
    "feedback": (_cmd_feedback, "Run 25-year feedback loop simulation"),
    "evaluate": (_cmd_evaluate, "Run financial evaluation"),
    "validate": (_cmd_validate, "Run behavioral validation gate"),
    "decide": (_cmd_decide, "Generate decision package"),
    "sensitivity": (_cmd_sensitivity, "Run behavioral LHS sensitivity"),
    "uncertainty": (_cmd_uncertainty, "Run Monte Carlo uncertainty analysis"),
    "version": (_cmd_version, "Print version and environment info"),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="urbansim-apm",
        description="APM corridor evaluation pipeline for Greater Lafayette",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "subcommands:\n"
            + "\n".join(f"  {name:14s} {desc}" for name, (_, desc) in _COMMANDS.items())
            + "\n\nRun 'urbansim-apm <command> --help' for subcommand options."
        ),
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print version info and exit",
    )
    parser.add_argument(
        "command", nargs="?", choices=list(_COMMANDS.keys()),
        help="Subcommand to run",
    )
    parser.add_argument(
        "args", nargs=argparse.REMAINDER,
        help="Arguments passed to the subcommand",
    )

    args = parser.parse_args()

    if args.version and not args.command:
        return _cmd_version([])

    if not args.command:
        parser.print_help()
        return 0

    handler, _ = _COMMANDS[args.command]
    try:
        return handler(args.args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    except KeyboardInterrupt:
        logger.info("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
