from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional

from .config import SDKConfig

# Keys of a case dict that are not part of the item's input.
_EXPECTED_KEY = "expected"
_ITEM_ID_KEY = "_dataset_item_id"


@dataclass
class Dataset:
    """A named test set stored in Langfuse, read at one point in time.

    Iterate it like a list of case dicts (it can be passed straight to
    `run_experiment(..., dataset=...)`). `version` is a short fingerprint of the
    items' content: two experiments are comparable only when they ran on the same
    version. `snapshot` is when the items were read; pass it back to
    `load_dataset(name, version=...)` to get exactly the same items again.
    """

    name: str
    cases: list[dict]
    version: str
    snapshot: datetime
    description: Optional[str] = None
    _item_ids: dict[str, str] = field(default_factory=dict, repr=False)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def item_id(self, case: dict) -> Optional[str]:
        """The Langfuse dataset item a case came from."""
        return case.get(_ITEM_ID_KEY)


def fingerprint(cases: Iterable[dict]) -> str:
    """Short, order-independent hash of the cases' content (ids, inputs, expected outputs)."""
    canonical = sorted(
        json.dumps({k: v for k, v in case.items() if k != _ITEM_ID_KEY}, sort_keys=True, default=str)
        for case in cases
    )
    return hashlib.sha256("\n".join(canonical).encode()).hexdigest()[:12]


class _LangfuseApi:
    """Minimal Langfuse public-API client for datasets.

    Plain REST rather than the Langfuse SDK's typed models: the SDK's models change
    between releases (4.x requires response fields a 3.x server doesn't send), and the
    REST endpoints are stable across server versions.
    """

    def __init__(self, config: Optional[SDKConfig]) -> None:
        config = config or SDKConfig.from_env()
        if not (config.langfuse_public_key and config.langfuse_secret_key):
            raise RuntimeError(
                "Datasets are stored in Langfuse: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
                "(the same pair ExperimentLens uses)."
            )
        self.base = config.langfuse_host.rstrip("/")
        token = base64.b64encode(f"{config.langfuse_public_key}:{config.langfuse_secret_key}".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}

    def request(self, method: str, path: str, body: Optional[dict] = None, query: Optional[dict] = None) -> Any:
        url = f"{self.base}{path}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
        data = json.dumps(body, default=str).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Langfuse {method} {path} failed ({exc.code}): {exc.read().decode()[:300]}") from None
        return json.loads(payload) if payload else None


def _client(config: Optional[SDKConfig]) -> _LangfuseApi:
    return _LangfuseApi(config)


def _item_id(dataset_name: str, case_id: Any) -> str:
    # Deterministic, so re-uploading a case updates its item instead of duplicating it.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"experimentlens-dataset:{dataset_name}:{case_id}"))


def _to_case(item: dict) -> dict:
    raw = item.get("input")
    data = raw if isinstance(raw, dict) else {"input": raw}
    case = {"id": item["id"], **data}
    if item.get("expectedOutput") is not None:
        case.setdefault(_EXPECTED_KEY, item["expectedOutput"])
    case[_ITEM_ID_KEY] = item["id"]
    return case


def upload_dataset(
    name: str,
    cases: Iterable[dict],
    description: Optional[str] = None,
    case_id_key: str = "id",
    config: Optional[SDKConfig] = None,
) -> Dataset:
    """Create the dataset if needed and add or update one item per case, e.g.

        upload_dataset("capitals", [{"id": "fr", "question": "Capital of France?", "expected": "Paris"}])

    Each case's `expected` value becomes the item's expected output and the rest its
    input. Items are keyed by `case_id_key`, so uploading the same case again updates
    it rather than adding a duplicate. Returns the dataset as it is now.
    """
    api = _client(config)
    try:
        api.request("POST", "/api/public/v2/datasets", {"name": name, "description": description, "metadata": {"created_by": "experimentlens"}})
    except RuntimeError:
        pass  # Already exists; Langfuse keeps the original.

    for case in cases:
        if case_id_key not in case:
            raise ValueError(f"Every case needs an {case_id_key!r} key to identify its dataset item")
        data = {k: v for k, v in case.items() if k not in (_EXPECTED_KEY, _ITEM_ID_KEY)}
        api.request("POST", "/api/public/dataset-items", {
            "datasetName": name,
            "id": _item_id(name, case[case_id_key]),
            "input": data,
            "expectedOutput": case.get(_EXPECTED_KEY),
        })
    return load_dataset(name, config=config)


def load_dataset(name: str, version: Optional[datetime] = None, config: Optional[SDKConfig] = None) -> Dataset:
    """Read a dataset's items, as they are now or as they were at `version` (a UTC
    datetime, such as an earlier Dataset's `snapshot`)."""
    if version is not None and version.tzinfo is None:
        raise ValueError("version must be timezone-aware (UTC), e.g. dataset.snapshot")
    snapshot = version or datetime.now(timezone.utc)
    api = _client(config)
    meta = api.request("GET", f"/api/public/v2/datasets/{urllib.parse.quote(name, safe='')}")
    items: list[dict] = []
    page = 1
    while True:
        response = api.request("GET", "/api/public/dataset-items", query={
            "datasetName": name, "page": page, "limit": 100, "version": snapshot.isoformat().replace("+00:00", "Z"),
        })
        items += response["data"]
        if page >= response["meta"]["totalPages"]:
            break
        page += 1
    cases = [_to_case(item) for item in items if item.get("status", "ACTIVE") == "ACTIVE"]
    return Dataset(
        name=name,
        cases=cases,
        version=fingerprint(cases),
        snapshot=snapshot,
        description=meta.get("description") if meta else None,
    )


def link_run_to_item(config: SDKConfig, run_name: str, item_id: str, trace_id: str, metadata: Optional[dict] = None) -> None:
    """Record in Langfuse that `trace_id` was produced for this dataset item, under a
    dataset run named `run_name`. Best effort: a failure only warns."""
    try:
        _client(config).request("POST", "/api/public/dataset-run-items", {
            "runName": run_name, "datasetItemId": item_id, "traceId": trace_id, "metadata": metadata or {},
        })
    except Exception as exc:  # pragma: no cover - network
        warnings.warn(f"Could not link trace {trace_id} to dataset item {item_id}: {exc}")
