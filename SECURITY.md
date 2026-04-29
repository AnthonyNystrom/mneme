# Security policy

## Supported versions

Only the latest **1.x** release of `mneme-cache` receives security fixes. Public surface is locked at v1.0; minor releases are additive and backward-compatible, so upgrading within 1.x should be safe.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security reports.

Email **nystrom.anthony@gmail.com** with:

- A description of the issue and the impact you observed
- Reproduction steps (or a minimal proof-of-concept)
- The `mneme-cache` version, Python version, and any relevant extras (`hnsw`, `redis`, `postgres`, `dynamodb`)
- Whether you'd like to be credited in the fix's release notes

You can expect:

- An acknowledgement within **3 business days**
- A triage decision (accepted / not-a-vuln / duplicate) within **7 business days**
- A fix or mitigation timeline once severity is agreed; we aim for a patch release within **30 days** for high-severity issues

## Scope

In scope:

- The published `mneme-cache` package on PyPI
- Anything in `src/mneme/` and the documented public API
- The store backends (`SQLiteStore`, `RedisStore`, `PostgresStore`, `DynamoDBStore`, `MemoryStore`)
- The calibration and migration tools (`mneme.tools.*`)

Out of scope:

- The `examples/showcase/` Flask demo (illustrative only; not a supported service)
- Vulnerabilities in optional third-party dependencies — please report those upstream
- Issues that require a malicious local user with filesystem access (the cache trusts its own SQLite file)
