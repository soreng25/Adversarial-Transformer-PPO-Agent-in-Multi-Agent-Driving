"""Immutable GPUDrive pin loading and source-tree verification."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
import configparser
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class PinError(RuntimeError):
    """Raised when immutable pin data or a pinned checkout is invalid."""


def repository_root(start: Path | None = None) -> Path:
    """Locate the port repository without relying on the process cwd."""

    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/runtime/gpudrive-pins.json").is_file():
            return candidate
    raise PinError("could not locate configs/runtime/gpudrive-pins.json")


def default_pin_path() -> Path:
    return repository_root() / "configs/runtime/gpudrive-pins.json"


def default_smoke_config_path() -> Path:
    return repository_root() / "configs/smoke/scene.json"


def _require_sha(value: Any, length: int, field: str) -> str:
    if not isinstance(value, str):
        raise PinError(f"{field} must be a string")
    pattern = SHA256_RE if length == 64 else GIT_SHA_RE
    if not pattern.fullmatch(value):
        raise PinError(f"{field} must be a lowercase {length}-character hex digest")
    return value


def load_pins(path: Path | str | None = None) -> dict[str, Any]:
    """Load and structurally validate the immutable runtime pin file."""

    pin_path = Path(path) if path is not None else default_pin_path()
    try:
        pins = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PinError(f"unable to load pin file {pin_path}: {exc}") from exc

    if pins.get("schema") != "gpudrive_runtime_pins" or pins.get(
        "schema_version"
    ) != 1:
        raise PinError("unsupported GPUDrive pin schema")

    gpudrive = pins.get("gpudrive")
    runtime = pins.get("reference_runtime")
    scene = pins.get("smoke_scene")
    if not all(isinstance(value, dict) for value in (gpudrive, runtime, scene)):
        raise PinError("pin file is missing gpudrive, reference_runtime, or smoke_scene")

    _require_sha(gpudrive.get("commit"), 40, "gpudrive.commit")
    _require_sha(gpudrive.get("tree"), 40, "gpudrive.tree")
    _require_sha(gpudrive.get("uv_lock_sha256"), 64, "gpudrive.uv_lock_sha256")
    submodules = gpudrive.get("submodules")
    if not isinstance(submodules, dict) or not submodules:
        raise PinError("gpudrive.submodules must be a non-empty object")
    for name, revision in submodules.items():
        _require_sha(revision, 40, f"gpudrive.submodules[{name!r}]")

    _require_sha(scene.get("sha256"), 64, "smoke_scene.sha256")
    for digest_field in ("base_image_index_digest", "base_image_amd64_digest"):
        digest = runtime.get(digest_field)
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise PinError(f"reference_runtime.{digest_field} must use sha256:<hex>")
        _require_sha(digest.removeprefix("sha256:"), 64, digest_field)
    uv = runtime.get("uv")
    if not isinstance(uv, dict):
        raise PinError("reference_runtime.uv must be an object")
    _require_sha(uv.get("wheel_sha256"), 64, "reference_runtime.uv.wheel_sha256")
    return pins


def load_smoke_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else default_smoke_config_path()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PinError(f"unable to load smoke config {config_path}: {exc}") from exc
    if config.get("schema") != "gpudrive_scene_smoke" or config.get(
        "schema_version"
    ) != 1:
        raise PinError("unsupported scene-smoke config schema")
    if config.get("failure_definition") is not None:
        raise PinError("Milestone A smoke config must not define research failure")
    environment = config.get("environment", {})
    if environment.get("num_worlds") != 1 or environment.get("max_cont_agents") != 1:
        raise PinError("Milestone A smoke must use one world and one controlled agent")
    action = config.get("action", {})
    commands = action.get("commands")
    if not isinstance(commands, list) or not commands:
        raise PinError("Milestone A smoke action.commands must be non-empty")
    names: set[str] = set()
    for index, command in enumerate(commands):
        if not isinstance(command, dict) or not isinstance(command.get("name"), str):
            raise PinError(f"action.commands[{index}] must have a string name")
        if command["name"] in names:
            raise PinError(f"duplicate smoke command name: {command['name']}")
        names.add(command["name"])
        kind = command.get("kind")
        values = (
            command.get("expected_accel_steer_head")
            if kind == "discrete"
            else command.get("accel_steer_head")
        )
        if kind not in {"discrete", "physical"}:
            raise PinError(f"unsupported smoke command kind: {kind!r}")
        if (
            not isinstance(values, list)
            or len(values) != 3
            or not all(isinstance(value, (int, float)) for value in values)
        ):
            raise PinError(
                f"action.commands[{index}] must define three physical values"
            )
        if kind == "discrete" and not isinstance(command.get("index"), int):
            raise PinError(f"action.commands[{index}] must define an integer index")
    expected = config.get("expected", {})
    if expected.get("physical_action_order") != [
        "acceleration",
        "steering",
        "head_angle",
    ]:
        raise PinError("smoke physical action order must be acceleration/steering/head")
    return config


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tree_sha256(path: Path | str) -> str | None:
    """Hash a file directly, or a directory by names and file bytes."""

    root = Path(path)
    if not root.exists():
        return None
    if root.is_file():
        return sha256_file(root)
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _run_git(source: Path, *args: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = result.stdout.rstrip() if result.returncode == 0 else result.stderr.strip()
    return result.returncode == 0, output


def _recursive_submodule_checkouts(
    repository: Path,
    *,
    prefix: Path | None = None,
    visited: set[Path] | None = None,
) -> list[dict[str, Any]]:
    """Verify recursive checkouts against each parent tree without Git's shell helper."""

    logical_prefix = prefix or Path()
    seen = visited if visited is not None else set()
    resolved_repository = repository.resolve()
    if resolved_repository in seen:
        return [
            {
                "path": logical_prefix.as_posix(),
                "ok": False,
                "error": "recursive submodule cycle",
            }
        ]
    seen.add(resolved_repository)
    modules_path = repository / ".gitmodules"
    if not modules_path.is_file():
        return []
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with modules_path.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, configparser.Error, UnicodeError) as exc:
        return [
            {
                "path": logical_prefix.as_posix(),
                "ok": False,
                "error": f"cannot parse .gitmodules: {exc}",
            }
        ]

    rows: list[dict[str, Any]] = []
    for section in sorted(parser.sections()):
        raw_path = parser.get(section, "path", fallback="")
        relative = Path(raw_path)
        safe = bool(raw_path) and not relative.is_absolute() and ".." not in relative.parts
        logical_path = logical_prefix / relative if safe else logical_prefix
        checkout = repository / relative if safe else repository
        gitlink_ok, gitlink = (
            _run_git(repository, "rev-parse", f"HEAD:{relative.as_posix()}")
            if safe
            else (False, "unsafe or missing submodule path")
        )
        initialized = safe and checkout.is_dir() and (checkout / ".git").exists()
        head_ok, checkout_head = (
            _run_git(checkout, "rev-parse", "HEAD")
            if initialized
            else (False, "missing or uninitialized")
        )
        row_ok = gitlink_ok and head_ok and gitlink == checkout_head
        rows.append(
            {
                "path": logical_path.as_posix(),
                "ok": row_ok,
                "gitlink": gitlink,
                "checkout_head": checkout_head,
            }
        )
        if row_ok:
            rows.extend(
                _recursive_submodule_checkouts(
                    checkout,
                    prefix=logical_path,
                    visited=seen,
                )
            )
    return rows


def _check(
    name: str,
    ok: bool,
    *,
    expected: Any = None,
    observed: Any = None,
    required: bool = True,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "required": required,
        "expected": expected,
        "observed": observed,
        "message": message,
    }


def verify_scene(scene_path: Path, scene_pin: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(_check("scene.exists", scene_path.is_file(), observed=str(scene_path)))
    if not scene_path.is_file():
        return checks

    observed_hash = sha256_file(scene_path)
    checks.append(
        _check(
            "scene.sha256",
            observed_hash == scene_pin["sha256"],
            expected=scene_pin["sha256"],
            observed=observed_hash,
        )
    )
    try:
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        sdc_index = int(scene["metadata"]["sdc_track_index"])
        sdc = scene["objects"][sdc_index]
    except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        checks.append(_check("scene.structure", False, observed=str(exc)))
        return checks

    checks.extend(
        [
            _check(
                "scene.scenario_id",
                scene.get("scenario_id") == scene_pin["scenario_id"],
                expected=scene_pin["scenario_id"],
                observed=scene.get("scenario_id"),
            ),
            _check(
                "scene.sdc_track_index",
                sdc_index == scene_pin["source_sdc_track_index"],
                expected=scene_pin["source_sdc_track_index"],
                observed=sdc_index,
            ),
            _check(
                "scene.sdc_object_id",
                sdc.get("id") == scene_pin["sdc_object_id"],
                expected=scene_pin["sdc_object_id"],
                observed=sdc.get("id"),
            ),
            _check("scene.sdc_type", sdc.get("type") == "vehicle", expected="vehicle", observed=sdc.get("type")),
            _check(
                "scene.sdc_mark_as_expert",
                sdc.get("mark_as_expert") is False,
                expected=False,
                observed=sdc.get("mark_as_expert"),
            ),
        ]
    )
    return checks


def verify_source_tree(
    source: Path | str,
    pins: dict[str, Any],
    *,
    require_initialized_submodules: bool = True,
) -> list[dict[str, Any]]:
    """Verify the exact source commit, gitlinks, lock, package, and smoke scene."""

    root = Path(source).resolve()
    gpudrive_pin = pins["gpudrive"]
    checks: list[dict[str, Any]] = [
        _check("source.exists", root.is_dir(), observed=str(root))
    ]
    if not root.is_dir():
        return checks

    ok, head = _run_git(root, "rev-parse", "HEAD")
    checks.append(
        _check(
            "source.commit",
            ok and head == gpudrive_pin["commit"],
            expected=gpudrive_pin["commit"],
            observed=head,
        )
    )
    ok, tree = _run_git(root, "rev-parse", "HEAD^{tree}")
    checks.append(
        _check(
            "source.tree",
            ok and tree == gpudrive_pin["tree"],
            expected=gpudrive_pin["tree"],
            observed=tree,
        )
    )
    ok, tracked_status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    )
    checks.append(
        _check(
            "source.tracked_worktree_clean",
            ok and tracked_status == "",
            expected="no tracked modifications or changed submodule checkouts",
            observed=tracked_status,
        )
    )
    ok, untracked_output = _run_git(
        root, "ls-files", "--others", "--exclude-standard"
    )
    untracked_files = untracked_output.splitlines() if ok else []
    allowed_untracked_prefixes = (".venv/", "build/", "gpudrive_cache/")
    unexpected_untracked = [
        path
        for path in untracked_files
        if not path.replace("\\", "/").startswith(allowed_untracked_prefixes)
    ]
    checks.append(
        _check(
            "source.untracked_files_allowed",
            ok and not unexpected_untracked,
            expected={"allowed_prefixes": list(allowed_untracked_prefixes)},
            observed={
                "unexpected": unexpected_untracked,
                "total_untracked": len(untracked_files),
                "command_error": None if ok else untracked_output,
            },
        )
    )
    ok, remote = _run_git(root, "config", "--get", "remote.origin.url")
    normalized_remote = remote.removesuffix("/").removesuffix(".git")
    normalized_expected = gpudrive_pin["repository"].removesuffix("/").removesuffix(".git")
    checks.append(
        _check(
            "source.remote",
            ok and normalized_remote.lower() == normalized_expected.lower(),
            expected=gpudrive_pin["repository"],
            observed=remote,
        )
    )

    for relative, expected_revision in sorted(gpudrive_pin["submodules"].items()):
        ok, gitlink = _run_git(root, "rev-parse", f"HEAD:{relative}")
        checks.append(
            _check(
                f"source.gitlink.{relative}",
                ok and gitlink == expected_revision,
                expected=expected_revision,
                observed=gitlink,
            )
        )
        checkout = root / relative
        checkout_is_initialized = checkout.is_dir() and (checkout / ".git").exists()
        ok, checkout_head = (
            _run_git(checkout, "rev-parse", "HEAD")
            if checkout_is_initialized
            else (False, "missing or uninitialized")
        )
        checks.append(
            _check(
                f"source.submodule_checkout.{relative}",
                ok and checkout_head == expected_revision,
                expected=expected_revision,
                observed=checkout_head,
                required=require_initialized_submodules,
                message=None if require_initialized_submodules else "gitlink verification is authoritative for this source-only check",
            )
        )

    recursive_rows = _recursive_submodule_checkouts(root)
    invalid_recursive = [row for row in recursive_rows if not row["ok"]]
    checks.append(
        _check(
            "source.recursive_submodules_initialized",
            bool(recursive_rows) and not invalid_recursive,
            expected="every recursive checkout matches its parent-tree gitlink",
            observed={
                "count": len(recursive_rows),
                "invalid": invalid_recursive,
            },
            required=require_initialized_submodules,
            message=(
                None
                if require_initialized_submodules
                else "recursive checkout is optional for this source-only audit"
            ),
        )
    )

    lock_path = root / "uv.lock"
    lock_hash = sha256_file(lock_path) if lock_path.is_file() else None
    checks.append(
        _check(
            "source.uv_lock_sha256",
            lock_hash == gpudrive_pin["uv_lock_sha256"],
            expected=gpudrive_pin["uv_lock_sha256"],
            observed=lock_hash,
        )
    )

    package_version = None
    try:
        with (root / "pyproject.toml").open("rb") as handle:
            package_version = tomllib.load(handle)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        pass
    checks.append(
        _check(
            "source.package_version",
            package_version == gpudrive_pin["package_version"],
            expected=gpudrive_pin["package_version"],
            observed=package_version,
        )
    )

    scene_path = root / pins["smoke_scene"]["relative_path"]
    checks.extend(verify_scene(scene_path, pins["smoke_scene"]))
    return checks


def required_checks_pass(checks: list[dict[str, Any]]) -> bool:
    return all(check["ok"] for check in checks if check.get("required", True))
