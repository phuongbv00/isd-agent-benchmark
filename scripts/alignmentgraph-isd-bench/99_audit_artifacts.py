#!/usr/bin/env python3
"""Final gate: is every paper artifact present, real, fresh and wired in?

Runs last, after the whole pipeline (07 -> 09/10/13, ablation 08 -> 11/12, 14,
00_validate_bloom, then docs/scripts/sync_generated.sh). Checking that files
exist is the weakest possible read, so this audits the four ways an artifact
can be "there" and still be wrong:

  * a generator gained an output that the inventory, the guide and the paper
    never learned about, so nobody notices it is missing (A)
  * the file exists but is a PLACEHOLDER — the generators deliberately emit a
    valid, \\input-able comment-only file when the pooled layer they need is
    absent, so LaTeX compiles happily and the table is silently gone (D)
  * the docs/ copy is an older generation than the pooled JSON it names as
    its source: a partial regeneration, or sync_generated.sh never run (C, E).
    Both read the header line the generators write, never mtime — the sync
    step is a plain `cp`, which stamps every copied file with the time of the
    sync rather than the time of generation.
  * the file is fine but nothing references it, or the manuscript references a
    macro no generator defines — an undefined control sequence is a build
    failure that no amount of file-presence checking can see (G, H)

Checks, in report order:

  A  inventory drift    every filename a generator writes is declared below,
                        and every declaration maps to a real generator
  B  presence           required artifacts exist in results/generated/
  C  sync               docs/generated/ matches results/generated/ byte-wise,
                        and sync_generated.sh actually covers each artifact
  D  placeholders       no artifact is a placeholder / TBD-DEMO stub
  E  freshness          each artifact is at least as new as the pooled JSON it
                        names, and the batch agrees on one source generation
  F  figure pairing     every figure ships both .pdf (vector) and .png
  G  LaTeX references   \\input/\\includegraphics targets resolve; artifacts
                        nothing references are listed so the gap is visible
  H  macros             every generated-looking macro used in the manuscript
                        is defined by one of the generated macro files
  I  language           artifact text is English with a decimal point, so a
                        Vietnamese caption cannot creep back in unnoticed

What counts as required follows the pooled layers that exist: ablation
artifacts are expected once pooled_ablation.json is there, sensitivity ones
once the pooled ladder carries alignment_sensitivity, and the Bloom validity
notes are WARN-only because they need a manually trained checkpoint rather
than a pipeline step.

Usage:
  python scripts/alignmentgraph-isd-bench/99_audit_artifacts.py
  python scripts/alignmentgraph-isd-bench/99_audit_artifacts.py --strict   # WARN also fails
  python scripts/alignmentgraph-isd-bench/99_audit_artifacts.py --no-docs  # benchmark side only

Exit code 0 = no RED findings (and no WARN under --strict), 1 = otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]          # isd-agent-benchmark/
SRC_DIR = REPO_ROOT / "results" / "generated"
POOLED_LADDER = REPO_ROOT / "results" / "pooled_ladder.json"
POOLED_ABLATION = REPO_ROOT / "results" / "pooled_ablation.json"

THESIS_ROOT = REPO_ROOT.parent                            # master-thesis/
DOCS_DIR = THESIS_ROOT / "docs" / "generated"
SYNC_SH = THESIS_ROOT / "docs" / "scripts" / "sync_generated.sh"
#: Every manuscript that may consume the generated artifacts. Missing dirs are
#: skipped, so this stays correct if the submodule is checked out standalone.
MANUSCRIPT_DIRS = [THESIS_ROOT / "docs" / "paper", THESIS_ROOT / "docs" / "thesis"]

GEN_TABLES = "alignmentgraph-isd-bench/09_gen_paper_tables.py"
GEN_FIGURES = "alignmentgraph-isd-bench/10_gen_paper_figures.py"
GEN_COST = "alignmentgraph-isd-bench/13_gen_cost_table.py"
GEN_SENS = "alignmentgraph-isd-bench/14_gen_sensitivity_table.py"
GEN_ABL_TABLES = "alignmentgraph-isd-bench/ablation/11_gen_ablation_tables.py"
GEN_ABL_FIGURES = "alignmentgraph-isd-bench/ablation/12_gen_ablation_figures.py"
GEN_BLOOM = "alignmentgraph-isd-bench/00_validate_bloom.py"


@dataclass(frozen=True)
class Spec:
    """One expected artifact.

    ``name`` carries the extension for tables/notes and is a bare stem for
    figures, which always ship as a .pdf/.png pair (see check F).
    """

    name: str
    producer: str
    kind: str  # "tex" | "note" | "figure"
    need: str  # "always" | "ablation" | "sensitivity" | "bloom"
    #: False for artifacts whose filename never appears literally in their
    #: generator (00_validate_bloom writes wherever --out points), which would
    #: otherwise read as inventory drift in check A.
    scan: bool = True


#: The declared inventory. Check A holds this honest against the generators,
#: in both directions, so a new generator output cannot quietly go unnoticed
#: the way tab_rq1_total/tab_rq1_traj/tab_rq2_panel once did.
INVENTORY: list[Spec] = [
    # 09 — RQ1/RQ2 tables + prose macros. RQ1 leads on ADDIE and reports Total
    # and Trajectory alongside it; RQ2 leads on one signal and carries the full
    # 7-signal panel. Every one of those is a separate \input target.
    Spec("tab_rq1.tex", GEN_TABLES, "tex", "always"),
    Spec("tab_rq1_total.tex", GEN_TABLES, "tex", "always"),
    Spec("tab_rq1_traj.tex", GEN_TABLES, "tex", "always"),
    Spec("tab_rq2_alignment.tex", GEN_TABLES, "tex", "always"),
    Spec("tab_rq2_panel.tex", GEN_TABLES, "tex", "always"),
    Spec("stats_macros.tex", GEN_TABLES, "tex", "always"),
    Spec("stats_summary.md", GEN_TABLES, "note", "always"),
    # 10 — RQ1/RQ2 figures.
    Spec("fig_rq1_ladder", GEN_FIGURES, "figure", "always"),
    Spec("fig_rq1_delta", GEN_FIGURES, "figure", "always"),
    Spec("fig_rq1_components", GEN_FIGURES, "figure", "always"),
    Spec("fig_rq2_ladder", GEN_FIGURES, "figure", "always"),
    Spec("fig_rq2_components", GEN_FIGURES, "figure", "always"),
    Spec("fig_rq2_vs_rq1_scatter", GEN_FIGURES, "figure", "always"),
    Spec("fig_failure_rate", GEN_FIGURES, "figure", "always"),
    Spec("fig_cost_quality", GEN_FIGURES, "figure", "always"),
    # 13 — cost / deployability profile. "always": these read the ladder run
    # dirs directly rather than a pooled layer, so if there is a ladder at all
    # there is a cost table, and a missing one means the deployability claim in
    # the paper has no number under it. Freshness still keys on the pooled
    # ladder (source_stamp): both derive from the same run dirs, so a pooled
    # JSON newer than the cost artifacts means the run set moved on without
    # them. That is a conservative watermark, not an input dependency.
    Spec("tab_cost.tex", GEN_COST, "tex", "always"),
    Spec("cost_macros.tex", GEN_COST, "tex", "always"),
    Spec("stats_cost.md", GEN_COST, "note", "always"),
    # 11/12 — ablation, expected once pooled_ablation.json exists.
    Spec("tab_ablation.tex", GEN_ABL_TABLES, "tex", "ablation"),
    Spec("tab_ablation_factorial.tex", GEN_ABL_TABLES, "tex", "ablation"),
    Spec("tab_ablation_factorial_full.tex", GEN_ABL_TABLES, "tex", "ablation"),
    Spec("abl_macros.tex", GEN_ABL_TABLES, "tex", "ablation"),
    Spec("stats_ablation.md", GEN_ABL_TABLES, "note", "ablation"),
    Spec("fig_ablation", GEN_ABL_FIGURES, "figure", "ablation"),
    Spec("fig_ablation_ladder", GEN_ABL_FIGURES, "figure", "ablation"),
    Spec("fig_ablation_components", GEN_ABL_FIGURES, "figure", "ablation"),
    Spec("fig_ablation_factorial", GEN_ABL_FIGURES, "figure", "ablation"),
    # 14 — encoder sensitivity, expected once the pooled ladder carries it.
    Spec("tab_sensitivity.tex", GEN_SENS, "tex", "sensitivity"),
    Spec("sens_macros.tex", GEN_SENS, "tex", "sensitivity"),
    Spec("stats_sensitivity.md", GEN_SENS, "note", "sensitivity"),
    # 00 — Bloom validity evidence. Needs a hand-trained checkpoint, so its
    # absence is a WARN rather than a pipeline failure.
    Spec("bloom_validation.md", GEN_BLOOM, "note", "bloom", scan=False),
    Spec("bloom_arms_validation.md", GEN_BLOOM, "note", "bloom", scan=False),
]

#: Filenames written by a generator, as they appear in its ``outputs`` mapping.
#: The closing quote must follow the extension, which is what keeps the
#: header_comment("tab_rq2_panel.tex — placeholder") calls out of the match.
_SCAN_FILE = re.compile(r'"([a-z0-9_]+\.(?:tex|md))"')
#: save(fig, outdir, "fig_name", written) — also matches the GF.save() form
#: that the ablation figure script uses.
_SCAN_FIG = re.compile(r'save\(\s*fig\s*,\s*outdir\s*,\s*"([a-z0-9_]+)"')

#: Substrings that mean "this file rendered, but with nothing in it". Each one
#: is a literal the generators emit on a missing pooled layer or in --demo.
PLACEHOLDER_MARKS = [
    "TBD-DEMO",
    "— placeholder",
    "% No alignment_scores.json",
    "% No factorial layer",
    "— skipped.",
]

#: The generation stamp. The ``\bat\b`` is deliberate: the notes carry an inline
#: "AUTO-GENERATED by ... at <built> from pooled_x.json (generated_at <source>)"
#: header, and a greedy match would walk past the build stamp onto the source
#: one, making every note look exactly as old as its input.
_HDR_SELF = re.compile(r"^%?\s*(?:AUTO-GENERATED.*?\bat\b|generated_at:|Generated:)\s*"
                       r"(\d{4}-\d{2}-\d{2}T[\d:]+)", re.M)
_HDR_SOURCE = re.compile(r"(pooled_\w+\.json)\s*\(?generated_at:?\s*"
                         r"(\d{4}-\d{2}-\d{2}T[\d:]+)", re.M)
_INPUT_REF = re.compile(r"\\input\{[^}]*generated/([A-Za-z0-9_]+)\}")
_GRAPHIC_REF = re.compile(
    r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*generated/figures/([A-Za-z0-9_]+)")
_NEWCOMMAND = re.compile(r"\\newcommand\{\\([a-zA-Z]+)\}")
_MACRO_USE = re.compile(r"\\([a-z][a-zA-Z]{3,})")


@dataclass
class Report:
    red: list[str]
    warn: list[str]
    info: list[str]

    def __init__(self) -> None:
        self.red, self.warn, self.info = [], [], []

    def add(self, level: str, msg: str) -> None:
        getattr(self, level).append(msg)


def _ts(text: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def figure_files(spec: Spec, root: Path) -> list[Path]:
    return [root / "figures" / f"{spec.name}.{ext}" for ext in ("pdf", "png")]


def artifact_path(spec: Spec, root: Path) -> Path:
    """The path used for presence/sync: a figure is represented by its .pdf."""
    return figure_files(spec, root)[0] if spec.kind == "figure" else root / spec.name


def load_context() -> dict:
    """Which pooled layers exist — this decides what is required at all."""
    ctx: dict = {"ladder": None, "ablation": None, "sensitivity": False}
    if POOLED_LADDER.exists():
        pooled = json.loads(_read(POOLED_LADDER))
        ctx["ladder"] = _ts(pooled.get("generated_at"))
        ctx["sensitivity"] = bool(pooled.get("alignment_sensitivity"))
        ctx["ladder_demo"] = bool(pooled.get("_demo"))
    if POOLED_ABLATION.exists():
        ctx["ablation"] = _ts(json.loads(_read(POOLED_ABLATION)).get("generated_at"))
    return ctx


def level_for(spec: Spec, ctx: dict) -> str | None:
    """Severity of this artifact being absent, or None when not expected yet."""
    if spec.need == "always":
        return "red"
    if spec.need == "ablation":
        return "red" if ctx["ablation"] else None
    if spec.need == "sensitivity":
        return "red" if ctx["sensitivity"] else None
    return "warn"  # bloom


def source_stamp(spec: Spec, ctx: dict) -> datetime | None:
    """The pooled generation an artifact of this kind should be no older than."""
    return ctx["ablation"] if spec.producer.startswith(
        "alignmentgraph-isd-bench/ablation/") else ctx["ladder"]


#: Artifacts are English: captions and headers get read on slides and in a
#: venue's copy-edit pass, and a mixed-language float is the kind of thing that
#: survives to submission unnoticed. Vietnamese-only letters (the base Latin
#: alphabet is deliberately excluded) plus the VN decimal idiom `{,}`.
_VN_LETTERS = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯ]"
    r"|[àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]"
    r"|[ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ]")
_VN_DECIMAL = re.compile(r"\d\{,\}\d")


def check_language(rep: Report) -> None:
    """Every generated .tex/.md must be English with a decimal point."""
    if not DOCS_DIR.exists():
        return
    for path in sorted(DOCS_DIR.glob("*.tex")) + sorted(DOCS_DIR.glob("*.md")):
        text = _read(path)
        # Skip the AUTO-GENERATED provenance header: it names generator paths only.
        body = "\n".join(ln for ln in text.splitlines()
                         if not ln.startswith("% AUTO-GENERATED"))
        hits = sorted({m.group(0) for m in _VN_LETTERS.finditer(body)})
        if hits:
            rep.add("red", f"I: {path.name} still contains Vietnamese text "
                           f"(found {', '.join(hits[:8])}) — artifact text must be "
                           "English; fix the generator, not the artifact")
        if _VN_DECIMAL.search(body):
            rep.add("red", f"I: {path.name} uses the '{{,}}' decimal mark — "
                           "artifacts use a decimal point; check fmt_num and "
                           "size_display in the generator")


# ── A. inventory drift ───────────────────────────────────────────────────────

def check_inventory(rep: Report) -> None:
    declared_by_producer: dict[str, set[str]] = {}
    for spec in INVENTORY:
        if spec.scan:
            declared_by_producer.setdefault(spec.producer, set()).add(spec.name)

    for producer, declared in sorted(declared_by_producer.items()):
        script = REPO_ROOT / "scripts" / producer
        if not script.exists():
            rep.add("red", f"A: generator missing from disk: scripts/{producer}")
            continue
        src = _read(script)
        produced = set(_SCAN_FILE.findall(src)) | set(_SCAN_FIG.findall(src))
        for name in sorted(produced - declared):
            rep.add("red", f"A: scripts/{producer} writes {name!r} but this audit's "
                           "INVENTORY does not declare it — the docs and the paper "
                           "very likely do not know about it either")
        for name in sorted(declared - produced):
            rep.add("red", f"A: INVENTORY declares {name!r} from scripts/{producer}, "
                           "but that script no longer writes it")
    if not rep.red:
        rep.add("info", f"A: {len(INVENTORY)} declared artifacts match "
                        f"{len(declared_by_producer)} generators, both directions")


# ── B/D/E. presence, placeholders, freshness (benchmark side) ────────────────

def check_source(rep: Report, ctx: dict) -> list[Spec]:
    """Presence in results/generated/, plus placeholder and freshness reads.

    Returns the artifacts that actually exist, so later checks do not report a
    missing file twice under different headings.
    """
    present: list[Spec] = []
    source_seen: dict[str, set[str]] = {}

    for spec in INVENTORY:
        level = level_for(spec, ctx)
        path = artifact_path(spec, SRC_DIR)
        if not path.exists():
            if level:
                rep.add(level, f"B: missing {spec.name} "
                               f"(run scripts/{spec.producer})")
            continue
        present.append(spec)

        if spec.kind == "figure":
            # No header to read: fall back to the source-dir mtime, which is a
            # real generation time here (only the docs/ copies get restamped).
            stamp = datetime.fromtimestamp(path.stat().st_mtime)
            expected = source_stamp(spec, ctx)
            if expected and stamp < expected:
                rep.add("red", f"E: {spec.name} was rendered {stamp:%Y-%m-%d %H:%M} "
                               f"but its pooled input is newer "
                               f"({expected:%Y-%m-%d %H:%M}) — re-run "
                               f"scripts/{spec.producer}")
            continue

        text = _read(path)
        for mark in PLACEHOLDER_MARKS:
            if mark in text:
                rep.add("red", f"D: {spec.name} is a placeholder (matched {mark!r}) "
                               "— it will \\input cleanly and render nothing")
                break

        self_stamp = _ts(m.group(1) if (m := _HDR_SELF.search(text)) else None)
        src_match = _HDR_SOURCE.search(text)
        if src_match:
            source_seen.setdefault(src_match.group(1), set()).add(src_match.group(2))
            claimed = _ts(src_match.group(2))
            actual = source_stamp(spec, ctx)
            if claimed and actual and claimed < actual:
                rep.add("red", f"E: {spec.name} was built from an older "
                               f"{src_match.group(1)} ({claimed:%Y-%m-%d %H:%M}) than "
                               f"the one on disk ({actual:%Y-%m-%d %H:%M}) — re-run "
                               f"scripts/{spec.producer}")
        if self_stamp and (expected := source_stamp(spec, ctx)) and self_stamp < expected:
            rep.add("red", f"E: {spec.name} predates its pooled input — re-run "
                           f"scripts/{spec.producer}")

    if ctx.get("ladder_demo"):
        rep.add("red", "D: pooled_ladder.json carries _demo — every number "
                       "downstream of it is a fake placeholder")
    for pooled, stamps in sorted(source_seen.items()):
        if len(stamps) > 1:
            rep.add("red", f"E: artifacts disagree on which {pooled} they came from "
                           f"({', '.join(sorted(stamps))}) — partial regeneration, "
                           "re-run the generators for that layer together")
    return present


# ── C. sync into docs/generated/ ─────────────────────────────────────────────

def check_sync(rep: Report, present: list[Spec]) -> None:
    if not DOCS_DIR.exists():
        rep.add("info", f"C: no {DOCS_DIR} — docs-side checks skipped")
        return

    sync_src = _read(SYNC_SH) if SYNC_SH.exists() else ""
    if not sync_src:
        rep.add("warn", f"C: {SYNC_SH.name} not found — cannot verify copy coverage")

    for spec in present:
        for src in ([artifact_path(spec, SRC_DIR)] if spec.kind != "figure"
                    else figure_files(spec, SRC_DIR)):
            if not src.exists():
                continue
            dst = DOCS_DIR / src.relative_to(SRC_DIR)
            if not dst.exists():
                rep.add("red", f"C: {src.name} was generated but never synced into "
                               "docs/generated/ — run docs/scripts/sync_generated.sh")
            elif dst.read_bytes() != src.read_bytes():
                rep.add("red", f"C: docs/generated/{src.name} differs from the "
                               "generated copy — stale, run "
                               "docs/scripts/sync_generated.sh")
        # .tex is covered by a blanket glob and figures by per-extension globs,
        # but every .md is named one at a time, so a new note silently misses.
        if spec.kind == "note" and sync_src and spec.name not in sync_src:
            rep.add("red", f"C: {spec.name} is not named in sync_generated.sh — it "
                           "will never reach docs/generated/ (the script copies "
                           "*.tex wholesale but .md files by name)")

    orphans = sorted(
        p.name for p in DOCS_DIR.glob("*")
        if p.is_file() and not (SRC_DIR / p.name).exists()
    ) + sorted(
        f"figures/{p.name}" for p in (DOCS_DIR / "figures").glob("*")
        if p.is_file() and not (SRC_DIR / "figures" / p.name).exists()
    )
    for name in orphans:
        rep.add("warn", f"C: docs/generated/{name} has no source in "
                        "results/generated/ — it cannot be regenerated")


# ── F. figure pairing ────────────────────────────────────────────────────────

def check_figure_pairs(rep: Report, present: list[Spec]) -> None:
    for spec in present:
        if spec.kind != "figure":
            continue
        missing = [p.suffix.lstrip(".") for p in figure_files(spec, SRC_DIR)
                   if not p.exists()]
        if missing:
            rep.add("red", f"F: {spec.name} is missing its "
                           f"{'/'.join(missing)} form (LaTeX takes the vector .pdf, "
                           "the .png is the preview) — re-run "
                           f"scripts/{spec.producer}")


# ── G/H. manuscript wiring ───────────────────────────────────────────────────

def manuscript_files() -> list[Path]:
    return [p for root in MANUSCRIPT_DIRS if root.exists()
            for p in sorted(root.rglob("*.tex"))]


def check_references(rep: Report, present: list[Spec]) -> None:
    texs = manuscript_files()
    if not texs:
        rep.add("info", "G: no manuscript .tex found — reference checks skipped")
        return

    inputs: dict[str, list[Path]] = {}
    graphics: dict[str, list[Path]] = {}
    for path in texs:
        body = _read(path)
        for name in _INPUT_REF.findall(body):
            inputs.setdefault(name, []).append(path)
        for name in _GRAPHIC_REF.findall(body):
            graphics.setdefault(name, []).append(path)

    for name, users in sorted(inputs.items()):
        if not (DOCS_DIR / f"{name}.tex").exists():
            where = ", ".join(str(p.relative_to(THESIS_ROOT)) for p in users)
            rep.add("red", f"G: \\input{{generated/{name}}} in {where} has no "
                           f"docs/generated/{name}.tex — the build will fail")
    for name, users in sorted(graphics.items()):
        if not any((DOCS_DIR / "figures" / f"{name}.{ext}").exists()
                   for ext in ("pdf", "png")):
            where = ", ".join(str(p.relative_to(THESIS_ROOT)) for p in users)
            rep.add("red", f"G: \\includegraphics of generated/figures/{name} in "
                           f"{where} has no file — the build will fail")

    referenced = set(inputs) | set(graphics)
    dangling = [s.name for s in present
                if s.kind != "note" and s.name.removesuffix(".tex") not in referenced]
    if dangling:
        rep.add("warn", f"G: {len(dangling)} generated artifact(s) nothing references "
                        f"yet: {', '.join(sorted(dangling))}")


def check_macros(rep: Report) -> None:
    """Undefined generated macros are the one failure a file audit cannot see.

    Restricting the sweep to macro names whose leading lowercase run matches a
    generated family (ladder*, rqOne*, abl*, sens*) keeps LaTeX and package
    control sequences out of it without maintaining an allow-list.
    """
    texs = manuscript_files()
    if not texs or not DOCS_DIR.exists():
        return

    defined: dict[str, str] = {}
    for macro_file in sorted(DOCS_DIR.glob("*macros.tex")):
        for name in _NEWCOMMAND.findall(_read(macro_file)):
            defined[name] = macro_file.name
    if not defined:
        rep.add("red", "H: no \\newcommand found in docs/generated/*macros.tex — "
                       "every generated number in the prose is undefined")
        return

    families = {re.match(r"[a-z]+", name).group(0) for name in defined}
    used: dict[str, set[str]] = {}
    for path in texs:
        for name in _MACRO_USE.findall(_read(path)):
            if name != name.lower() and re.match(r"[a-z]+", name).group(0) in families:
                used.setdefault(name, set()).add(str(path.relative_to(THESIS_ROOT)))

    for name in sorted(set(used) - set(defined)):
        rep.add("red", f"H: \\{name} is used in {', '.join(sorted(used[name]))} but no "
                       "generated macro file defines it — LaTeX will abort with "
                       "'Undefined control sequence'")
    rep.add("info", f"H: {len(defined)} macros defined, {len(used)} used in the "
                    f"manuscript, families {'/'.join(sorted(families))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true",
                        help="Treat WARN findings as failures too.")
    parser.add_argument("--no-docs", action="store_true",
                        help="Audit only the benchmark side (skip docs/ sync, "
                             "LaTeX references and macros).")
    args = parser.parse_args()

    if not SRC_DIR.exists():
        print(f"ERROR: {SRC_DIR} not found — run the generator scripts first.",
              file=sys.stderr)
        return 1

    ctx = load_context()
    rep = Report()
    if not ctx["ladder"]:
        rep.add("red", f"B: {POOLED_LADDER.name} not found — nothing downstream of "
                       "07_pool_ladder_runs.py can be trusted")
    if not ctx["ablation"]:
        rep.add("info", "B: no pooled_ablation.json — ablation artifacts not expected")
    if not ctx["sensitivity"]:
        rep.add("info", "B: pooled ladder carries no alignment_sensitivity — "
                        "sensitivity artifacts not expected")

    check_inventory(rep)
    present = check_source(rep, ctx)
    check_figure_pairs(rep, present)
    if not args.no_docs:
        check_sync(rep, present)
        check_references(rep, present)
        check_macros(rep)
        check_language(rep)

    expected = sum(1 for s in INVENTORY if level_for(s, ctx))
    verdict = "OK" if not rep.red and not (args.strict and rep.warn) else "FAIL"
    print(f"== artifact audit [{verdict}] — {len(present)}/{expected} expected "
          f"artifacts present ==\n")
    for line in rep.red:
        print(f"  RED : {line}")
    for line in rep.warn:
        print(f"  WARN: {line}")
    for line in rep.info:
        print(f"  info: {line}")

    print(f"\n== {len(rep.red)} red, {len(rep.warn)} warn, "
          f"{len(rep.info)} info ==")
    return 1 if rep.red or (args.strict and rep.warn) else 0


if __name__ == "__main__":
    sys.exit(main())
