"""Dev helper (not used at runtime). Usage: python3 tools/dev/fix_whitelists.py <release folder>
Append the new release to NEW first. Extend version whitelists in V3/V4 tests: every tuple of "Vx.y" literals on a line that mentions VERSION gets the
V4.x releases appended (test logic untouched)."""
import re, sys, glob
NEW = ["V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3", "V4.8.4", "V4.8.5"]
TUP = re.compile(r'\((\s*"V\d+\.\d+(?:\.\d+)?"\s*(?:,\s*"V\d+\.\d+(?:\.\d+)?"\s*)*,?\s*)\)')
n_files = n_tuples = 0
for p in sorted(glob.glob(sys.argv[1] + "/tests/test_*.py")):
    src = open(p, encoding="utf-8").read()
    out, changed = [], False
    for line in src.split("\n"):
        if "VERSION" in line and "startswith(" not in line:
            def fix(m):
                global n_tuples
                items = re.findall(r'"(V\d+\.\d+(?:\.\d+)?)"', m.group(1))
                if not any(i.startswith("V3.") for i in items):
                    return m.group(0)
                add = [v for v in NEW if v not in items]
                if not add:
                    return m.group(0)
                n_tuples += 1
                return "(" + ", ".join(f'"{i}"' for i in items + add) + ")"
            new = TUP.sub(fix, line)
            changed |= new != line
            line = new
        out.append(line)
    if changed:
        n_files += 1
        open(p, "w", encoding="utf-8").write("\n".join(out))
print(f"{n_files} files, {n_tuples} whitelists extended")
