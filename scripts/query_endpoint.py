"""Send local images to the serving endpoint and print predictions + round-trip latency.

    python scripts/query_endpoint.py --profile <profile> --endpoint cv-image-classifier-dev img1.jpg img2.jpg
"""

import argparse
import base64
import time

from databricks.sdk import WorkspaceClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--profile", required=True, help="Databricks CLI profile")
    ap.add_argument("--endpoint", default="cv-image-classifier-dev")
    args = ap.parse_args()

    w = WorkspaceClient(profile=args.profile)
    for path in args.images:
        with open(path, "rb") as f:
            record = {"image": base64.b64encode(f.read()).decode()}
        t0 = time.perf_counter()
        pred = w.serving_endpoints.query(args.endpoint, dataframe_records=[record]).predictions[0]
        ms = (time.perf_counter() - t0) * 1000
        print(f"{path}: {pred['label']} ({pred['confidence']:.3f})  top_k={pred['top_k']}  {ms:.0f} ms")


if __name__ == "__main__":
    main()
