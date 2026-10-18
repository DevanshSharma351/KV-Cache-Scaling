"""Private fresh-process worker entry; revalidates the real remote receipt."""

import argparse
import json
from pathlib import Path

from .probe import ProbeRequest, run_probe, write_probe_result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    request = ProbeRequest.from_dict(json.loads(Path(args.request).read_text()))
    try:
        result = run_probe(request, root=args.root, receipt_path=args.receipt, run_id=args.run_id)
    except Exception as error:
        result = {"schema_version": "primary-probe-v1", "probe_id": request.probe_id,
                  "status": "authorization_error", "stage": "receipt", "length": request.length,
                  "error": {"type": type(error).__name__, "message": str(error)},
                  "peak_allocated_bytes": None, "peak_reserved_bytes": None}
    write_probe_result(args.output, result)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
