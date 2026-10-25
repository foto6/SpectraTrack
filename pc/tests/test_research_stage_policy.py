import copy
import json
from pathlib import Path

import pytest

from spectratrack.research_stage_policy import (
    CHARACTERIZATION_STAGE,
    EXTERNAL_STAGE,
    HISTORY_SCHEMA,
    POLICY_SCHEMA,
    RESULT_SCHEMA,
    STOP_STAGE,
    main,
    validate_history,
)

ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "benchmarks" / "vnext" / "smoke_stats" / "research_stage_policy.v1.json"
FIXTURE_PATH = ROOT / "benchmarks" / "vnext" / "smoke_stats" / "fixtures" / "research_stage_legal_history.json"


def _policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))

