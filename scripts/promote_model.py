"""Promote a registered model version to `champion`, or roll back.

    uv run python scripts/promote_model.py --candidate-version 5     # compare, then promote
    uv run python scripts/promote_model.py --candidate-version 5 --force
    uv run python scripts/promote_model.py --rollback

Rule: the candidate becomes champion only if its GBG500 metric (`gbg_reckless_auc`,
logged on its run) is strictly better than the champion's, or if there is no
champion yet. A champion without the metric (e.g. the dummy model) is replaced
by any candidate that has it. Every move tags the promoted version with
`promoted_at` and `previous_champion`, which `--rollback` uses.

Prints one JSON object on stdout. Exit codes: 0 done, 1 error, 2 promotion refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import mlflow
from mlflow.tracking import MlflowClient

from gbg_mlops.config import get_settings
from gbg_mlops.model_loader import EVALUATION_METRIC

EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2


@dataclass
class Decision:
    """Outcome of a promotion or rollback, printed as JSON."""

    action: str  # "promote" | "rollback"
    status: str  # "promoted" | "refused" | "unchanged" | "rolled_back"
    model_name: str
    alias: str
    champion_before: str | None
    champion_after: str | None
    candidate_version: str | None = None
    metric: str = EVALUATION_METRIC
    candidate_metric: float | None = None
    champion_metric: float | None = None
    forced: bool = False
    reason: str = ""


def _alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def _metric(client: MlflowClient, name: str, version: str) -> float | None:
    """Evaluation metric logged on the run that produced this version."""
    run_id = client.get_model_version(name, version).run_id
    if run_id is None:
        return None
    return client.get_run(run_id).data.metrics.get(EVALUATION_METRIC)


def _move_alias(client: MlflowClient, name: str, alias: str, version: str, previous: str | None):
    """Point the alias to `version` and record where it came from."""
    client.set_registered_model_alias(name, alias, version)
    client.set_model_version_tag(name, version, "promoted_at", datetime.now(UTC).isoformat())
    client.set_model_version_tag(name, version, "previous_champion", previous or "none")


def promote(
    client: MlflowClient, name: str, alias: str, candidate: str, force: bool = False
) -> Decision:
    """Move `alias` to `candidate` if it beats the current champion (or if forced)."""
    candidate = str(candidate)
    client.get_model_version(name, candidate)  # raises if the version does not exist
    champion = _alias_version(client, name, alias)
    decision = Decision(
        action="promote",
        status="refused",
        model_name=name,
        alias=alias,
        champion_before=champion,
        champion_after=champion,
        candidate_version=candidate,
        candidate_metric=_metric(client, name, candidate),
        champion_metric=_metric(client, name, champion) if champion else None,
        forced=force,
    )
    if champion == candidate:
        decision.status, decision.reason = "unchanged", f"version {candidate} is already {alias}"
        return decision

    if force:
        decision.reason = "forced, metrics not compared"
    elif champion is None:
        decision.reason = f"no {alias} yet"
    elif decision.candidate_metric is None:
        decision.reason = f"candidate has no '{EVALUATION_METRIC}' metric"
        return decision
    elif decision.champion_metric is None:
        decision.reason = f"current {alias} has no '{EVALUATION_METRIC}' metric"
    elif decision.candidate_metric > decision.champion_metric:
        decision.reason = (
            f"candidate {decision.candidate_metric:.4f} > {alias} {decision.champion_metric:.4f}"
        )
    else:
        decision.reason = (
            f"candidate {decision.candidate_metric:.4f} <= {alias} "
            f"{decision.champion_metric:.4f}: {alias} kept"
        )
        return decision

    _move_alias(client, name, alias, candidate, champion)
    decision.status, decision.champion_after = "promoted", candidate
    return decision


def rollback(client: MlflowClient, name: str, alias: str) -> Decision:
    """Point `alias` back to the version that was champion before the current one."""
    champion = _alias_version(client, name, alias)
    if champion is None:
        raise ValueError(f"no {alias} to roll back")
    previous = client.get_model_version(name, champion).tags.get("previous_champion")
    if previous in (None, "none"):
        raise ValueError(f"version {champion} has no previous {alias} to return to")
    client.get_model_version(name, previous)  # must still exist
    client.set_registered_model_alias(name, alias, previous)
    client.set_model_version_tag(name, champion, "rolled_back_at", datetime.now(UTC).isoformat())
    return Decision(
        action="rollback",
        status="rolled_back",
        model_name=name,
        alias=alias,
        champion_before=champion,
        champion_after=previous,
        reason=f"{alias} moved back from version {champion} to {previous}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--candidate-version", help="registered version to promote")
    action.add_argument("--rollback", action="store_true", help="restore the previous champion")
    parser.add_argument("--force", action="store_true", help="promote without comparing")
    args = parser.parse_args()
    if args.force and args.rollback:
        parser.error("--force only applies to --candidate-version")

    settings = get_settings()
    settings.export_mlflow_env()
    uri = settings.mlflow_tracking_uri
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    try:
        if args.rollback:
            decision = rollback(client, settings.model_name, settings.model_alias)
        else:
            decision = promote(
                client,
                settings.model_name,
                settings.model_alias,
                args.candidate_version,
                args.force,
            )
    except (ValueError, mlflow.exceptions.MlflowException) as error:
        print(json.dumps({"status": "error", "reason": str(error)}))
        return EXIT_ERROR
    print(json.dumps(asdict(decision), indent=2))
    return EXIT_OK if decision.status != "refused" else EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
