"""Export governance audit trail for reviewers."""

from datetime import datetime, timezone
from pathlib import Path

from agents.audit_logger import get_audit_log, format_audit_report

EXPORT_DIR = Path(__file__).resolve().parent.parent / "logs" / "exports"


def build_audit_report_text() -> str:
    entries = get_audit_log(limit=500)
    lines = [
        "=" * 60,
        "DRIFTWARDEN — GOVERNANCE AUDIT REPORT",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 60,
        "",
    ]
    for e in entries:
        lines.append(e.get("report") or format_audit_report(e))
        lines.append("-" * 40)
    return "\n".join(lines)


def save_audit_report() -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text(build_audit_report_text(), encoding="utf-8")
    return path
