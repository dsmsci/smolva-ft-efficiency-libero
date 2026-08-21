from __future__ import annotations

import subprocess
from pathlib import Path


def run(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    log_path: str | Path | None = None,
) -> None:
    print("$", " ".join(map(str, cmd)))
    if log_path is None:
        subprocess.run(cmd, cwd=cwd, check=True)
        return

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
