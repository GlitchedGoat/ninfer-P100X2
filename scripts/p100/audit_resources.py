#!/usr/bin/env python3
"""List GP100 resource risks in an sm_60 build: per-kernel static shared memory above the
48 KiB per-block limit, local-memory spills, and register pressure that caps occupancy.

Usage: scripts/p100/audit_resources.py build-p100/src/libninfer_ops.a [--all]
Dynamic shared memory is a launch-time value and is not visible here (see T-006).
"""
import re
import subprocess
import sys

SHARED_LIMIT = 48 * 1024


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    show_all = "--all" in sys.argv
    dump = subprocess.run(["cuobjdump", "--dump-resource-usage", sys.argv[1]],
                          check=True, capture_output=True, text=True).stdout
    rows = []
    name = None
    for line in dump.splitlines():
        match = re.search(r"Function (\S+):", line)
        if match:
            name = match.group(1)
            continue
        if name and "REG:" in line:
            fields = dict(re.findall(r"(\w+(?:\[\d+\])?):(\d+)", line))
            rows.append((name, int(fields.get("REG", 0)), int(fields.get("SHARED", 0)),
                         int(fields.get("LOCAL", 0)), int(fields.get("STACK", 0))))
            name = None
    names = subprocess.run(["c++filt"], input="\n".join(r[0] for r in rows),
                           capture_output=True, text=True).stdout.splitlines()
    flagged = 0
    for (mangled, reg, shared, local, stack), pretty in zip(rows, names):
        issues = []
        if shared > SHARED_LIMIT:
            issues.append(f"SHARED {shared} > 48 KiB")
        if local > 0:
            issues.append(f"LOCAL {local} B (spill)")
        if reg > 128:
            issues.append(f"REG {reg}")
        if issues or show_all:
            flagged += bool(issues)
            print(f"{'; '.join(issues) or 'ok':40s} reg={reg:3d} smem={shared:6d} "
                  f"local={local:5d} {pretty[:160]}")
    print(f"{len(rows)} kernels, {flagged} flagged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
