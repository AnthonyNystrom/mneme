"""DynamoDBStore quickstart.

This example uses moto (an in-process AWS mock) so it runs without a real
AWS account or DynamoDB Local container. Replace ``mock_aws()`` with your
real boto3 setup (IAM role, region, etc.) in production.

Install:  pip install "mneme[dynamodb]" "moto[dynamodb]"

Run:
    python examples/dynamodb_quickstart.py
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

# moto wants fake credentials in the environment.
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from mneme import SemanticCache
from mneme._store_dynamodb import DynamoDBStore


class ToyEmbedder:
    dim = 32
    fingerprint = "toy:hash:v1"

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = (digest * ((self.dim + len(digest) - 1) // len(digest)))[: self.dim]
        v = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32) - 128.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


def main() -> None:
    try:
        from moto import mock_aws
    except ImportError as exc:
        raise SystemExit(
            "This example requires moto. Install with:\n  pip install 'moto[dynamodb]'"
        ) from exc

    with mock_aws():
        # Auto-create the table on first open. In production you'd typically
        # provision the table out-of-band (CDK / Terraform) and pass
        # create_table=False.
        store = DynamoDBStore(
            table_name="my_app_cache",
            region_name="us-east-1",
            create_table=True,  # opt-in
            billing_mode="PAY_PER_REQUEST",  # default; no surprise capacity bills
        )
        with SemanticCache(store=store, embedder=ToyEmbedder()) as cache:
            cache.put("How do I reset my password?", "Click 'Forgot password' on login.")
            hit = cache.get("How do I reset my password?")
            assert hit is not None
            print(f"hit: {hit.layer}  {hit.response!r}")

            s = cache.stats()
            print(f"stats: entries={s.entries}  hits_exact={s.hits_exact}")


if __name__ == "__main__":
    main()
