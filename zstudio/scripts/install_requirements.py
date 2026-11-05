#!/usr/bin/env python3
"""Install zstudio's missing Python requirements into a private directory.

    install_requirements.py [--requirements FILE] [--target DIR] [--python EXE] [--strict]

Called by the CMake install step (see zstudio/CMakeLists.txt) and usable
by hand, e.g. for a source checkout:

    python3 zstudio/scripts/install_requirements.py   # -> zstudio/.python-deps

Each line of requirements.txt is ``<pip requirement>  # import: <module>``. A
requirement whose module the interpreter can import is left alone, whatever
provides it (distribution package, rosdep, pip). The missing ones are installed
with ``pip install --target --no-deps``, and so are, round by round, the
dependencies of what was installed that the interpreter does not already
satisfy. Nothing is written to the system Python (works with externally managed
interpreters, PEP 668) and nothing the interpreter already has is duplicated, so
the private directory never shadows or conflicts with system packages;
zstudio appends it to sys.path after them.

Exit status: 0 when every requirement is importable afterwards, or when
something failed without --strict (a warning names what is missing and the
command to run); 1 on failure with --strict.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUNDS = 8  # dependency depth; rosbags needs 2

# Runs in the studio's interpreter (isolated mode, the private directory last on
# sys.path, as at runtime). argv: target, JSON list of requirement strings to
# check in addition to the dependencies of every distribution in the target.
# Prints {"install": [...], "conflict": [[requirement, installed version], ...]}.
_CHECK = r"""
import json, sys
from importlib import metadata
try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
target = sys.argv[1]
sys.path.append(target)
found = {}
for dist in metadata.distributions():  # sys.path order: the first one wins, as on import
    name = canonicalize_name(dist.metadata["Name"] or "")
    if name and name not in found:
        found[name] = (dist.version, str(dist.locate_file("")).startswith(target))
wanted = [Requirement(r) for r in json.loads(sys.argv[2])]
for dist in metadata.distributions(path=[target]):
    wanted += [Requirement(r) for r in dist.requires or []]
install, conflict, seen = [], [], set()
for req in wanted:
    if req.marker is not None and not req.marker.evaluate({"extra": ""}):
        continue
    have = found.get(canonicalize_name(req.name))
    if have and (not req.specifier or req.specifier.contains(have[0], prereleases=True)):
        continue
    text = str(req.name) + str(req.specifier)
    if text in seen:
        continue
    seen.add(text)
    if have and not have[1]:
        conflict.append([text, have[0]])  # the interpreter's own copy would shadow ours
    else:
        install.append(text)
print(json.dumps({"install": install, "conflict": conflict}))
"""


def parse(path):
    """[(pip requirement, import module)] from a requirements file."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        spec, _, comment = line.partition("#")
        spec = spec.strip()
        if not spec:
            continue
        match = re.search(r"import:\s*([A-Za-z_][\w.]*)", comment)
        module = match.group(1) if match else re.split(r"[<>=!~;\[ ]", spec, 1)[0].replace("-", "_")
        out.append((spec, module))
    return out


def run_python(python, code, *args):
    return subprocess.run([python, "-I", "-c", code, *args], capture_output=True, text=True, timeout=120)


def importable(python, module, target):
    """Whether ``python`` (isolated from the user's environment) imports ``module``."""
    code = (
        "import importlib.util, sys\n"
        f"sys.path.append({str(target)!r})\n"
        f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
    )
    try:
        return run_python(python, code).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def unmet(python, target, requirements):
    """Requirements (plus the target's dependencies) the interpreter does not satisfy."""
    result = run_python(python, _CHECK, str(target), json.dumps(requirements))
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "dependency check failed")
    return json.loads(result.stdout)


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)  # interleave with pip's output in install logs
    root = HERE.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--requirements", default=str(root / "requirements.txt"))
    ap.add_argument("--target", default=str(root / ".python-deps"), help="private install directory")
    ap.add_argument("--python", default=sys.executable, help="interpreter zstudio runs with")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a requirement stays missing")
    args = ap.parse_args(argv)
    target = Path(args.target).expanduser().resolve()
    requirements = parse(args.requirements)
    missing = [(spec, module) for spec, module in requirements if not importable(args.python, module, target)]
    if not missing:
        print("zstudio: Python requirements satisfied (" + ", ".join(m for _, m in requirements) + ")")
        return 0

    print("zstudio: installing " + ", ".join(spec for spec, _ in missing) + " into " + str(target))
    pip = [args.python, "-m", "pip", "install", "--target", str(target), "--upgrade", "--no-deps",
           "--disable-pip-version-check", "--no-warn-script-location", "--no-input"]
    problems = []
    todo = [spec for spec, _ in missing]
    try:
        target.mkdir(parents=True, exist_ok=True)
        for _ in range(ROUNDS):
            if not todo:
                break
            if subprocess.run(pip + todo).returncode != 0:
                problems.append("pip install failed for " + ", ".join(todo))
                break
            state = unmet(args.python, target, [])
            for requirement, version in state["conflict"]:
                problems.append(f"{requirement} is required, but the interpreter has {version}")
            todo = state["install"]
        if todo and not problems:
            problems.append("dependencies still unresolved after %d rounds: %s" % (ROUNDS, ", ".join(todo)))
        if (target / "bin").is_dir():  # console scripts of dependencies: unused
            for item in (target / "bin").iterdir():
                item.unlink()
            (target / "bin").rmdir()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        problems.append(str(error))

    still = [module for spec, module in missing if not importable(args.python, module, target)]
    if not still and not problems:
        print("zstudio: Python requirements installed")
        return 0
    message = "WARNING zstudio: " + "; ".join(problems or ["install incomplete"])
    if still:
        message += (
            ".\n  Still missing: " + ", ".join(still) + ". zstudio runs without the optional ones"
            " (bags, report plots and OpenCV extras are then disabled).\n"
            "  Install them with your package manager, or: " + args.python + " -m pip install --target "
            + str(target) + " " + " ".join(spec for spec, module in missing if module in still)
        )
    print(message, file=sys.stderr)
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
