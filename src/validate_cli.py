"""CLI wrapper helpers around src.validate."""
import logging
from pathlib import Path
from typing import Dict, Iterable

from src import validate

logger = logging.getLogger(__name__)


def run(paths: Iterable[str], id_prop: str = "ZONE_ID") -> Dict[str, dict]:
    """Validate multiple files and return per-file status records."""
    if validate.gpd is None:
        raise RuntimeError("geopandas not available; cannot validate geojson")

    results: Dict[str, dict] = {}
    for raw_path in paths:
        p = str(Path(raw_path))
        try:
            validate.validate_geojson(p, id_prop=id_prop)
            results[p] = {"status": "ok"}
        except Exception as exc:
            results[p] = {
                "status": "error",
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
    return results


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Validate one or more spatial files.")
    parser.add_argument("paths", nargs="+", help="Input GeoJSON/Shapefile paths")
    parser.add_argument("--id-prop", default="ZONE_ID", help="Identifier property name")
    args = parser.parse_args()

    output = run(args.paths, id_prop=args.id_prop)
    logger.info(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
