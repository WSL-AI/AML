from pathlib import Path
import json
import uuid

def new_run(parent, group):
    """Allocate a unique output directory; never replace an existing experiment."""
    from datetime import datetime, timezone

    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12]
    path = Path(parent) / group / name
    path.mkdir(parents=True, exist_ok=False)
    return path

def set_active(parent, filename, path):
    """Publish a completed run pointer atomically."""
    parent = Path(parent)
    temp = parent / (filename + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps({"path": str(Path(path).resolve().relative_to(parent.resolve()))}))
    temp.replace(parent / filename)

def resolve_training(run):
    run = Path(run)
    pointer = run / "active_training.json"
    path = run / json.loads(pointer.read_text())["path"] if pointer.exists() else run
    if not (path / "champion.joblib").exists():
        raise FileNotFoundError("Run notebook 03 first")
    return path
