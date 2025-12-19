from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List
import re


def generate_stage_error_summary(
    status_dict: Dict[str, Dict], stage: str, report_file: str
):
    """Generate detailed error report for compilation failures, appending to existing report."""

    error_report_file = report_file.replace(".md", f"_{stage}_errors.md")
    # Filter for compilation failures only
    stage_failures = {
        name: data
        for name, data in status_dict.items()
        if data.get("exit_status") == stage and f"{stage}_status" in data
    }

    if not stage_failures:
        return

    # Count by compilation status
    timeout_count = sum(
        1
        for data in stage_failures.values()
        if data.get(f"{stage}_status") == "timeout"
    )
    error_count = sum(
        1 for data in stage_failures.values() if data.get(f"{stage}_status") == "error"
    )
    no_log_count = sum(
        1 for data in stage_failures.values() if data.get(f"{stage}_status") == "no_log"
    )

    with open(error_report_file, "a", encoding="utf-8") as f:
        # Write compilation error summary
        f.write("\n# COMPILATION ERROR ANALYSIS\n\n")
        f.write(f"**Total {stage} failures: {len(stage_failures)}**\n\n")
        f.write("| Status | Count |\n")
        f.write("|---|---|\n")
        f.write(f"| Timeouts | {timeout_count} |\n")
        f.write(f"| Errors | {error_count} |\n")
        f.write(f"| No log | {no_log_count} |\n\n")

        # ERROR TESTS (grouped by error)
        if error_count > 0:
            f.write("## ERRORS (grouped by error)\n\n")
            # Group tests by error signature
            error_groups: Dict[str, List] = {}
            for name, data in stage_failures.items():
                if data.get(f"{stage}_status") == "error":
                    error_key = data.get("error_signature", "") or "(no error details)"
                    if error_key not in error_groups:
                        error_groups[error_key] = []
                    error_groups[error_key].append((name, data))

            # Summary table sorted by count
            f.write("| Error | Count |\n")
            f.write("|---|---|\n")
            for error_signature, tests in sorted(
                error_groups.items(), key=lambda x: len(x[1]), reverse=True
            ):
                error_escaped = error_signature.replace("|", "\\|").replace("\n", " ")
                f.write(f"| {error_escaped} | {len(tests)} |\n")
            f.write("\n")

            f.write("## ERROR TESTS (grouped by error)\n\n")

            # Detailed tables grouped by error
            for error_signature, tests in sorted(
                error_groups.items(), key=lambda x: len(x[1]), reverse=True
            ):
                error_escaped = error_signature.replace("|", "\\|").replace("\n", " ")
                f.write(f"### Error: {error_escaped}\n\n")
                f.write(f"**Count: {len(tests)}**\n\n")
                f.write("| test_name | command | error |\n")
                f.write("|---|---|---|\n")
                for name, data in tests:
                    command = (
                        data.get("command", "").replace("|", "\\|").replace("\n", " ")
                    )
                    error = data.get("error", "").replace("|", "\\|").replace("\n", " ")
                    f.write(f"| {name} | {command} | {error} |\n")
                f.write("\n")

        # TIMEOUT TESTS
        if timeout_count > 0:
            f.write("## TIMEOUT TESTS\n\n")
            f.write("| test_name | command |\n")
            f.write("|---|---|\n")
            for name, data in stage_failures.items():
                if data.get(f"{stage}_status") == "timeout":
                    command = (
                        data.get("command", "").replace("|", "\\|").replace("\n", " ")
                    )
                    f.write(f"| {name} | {command} |\n")
            f.write("\n")

        # NO LOG FILE
        if no_log_count > 0:
            f.write("## NO LOG FILE\n\n")
            f.write("| test_name |\n")
            f.write("|---|\n")
            for name, data in stage_failures.items():
                if data.get(f"{stage}_status") == "no_log":
                    f.write(f"| {name} |\n")
            f.write("\n")


class ErrorHandler(ABC):
    """Base class to handle error analysis and logging."""

    def __init__(self, log_dir: str, stage: str):
        """
        Initialize the handler with a log directory.

        Args:
            log_dir: Base directory containing the test logs
            stage: The stage name (e.g., 'compilation')
        """
        self.stage = stage
        self.log_path = Path(log_dir)
        self.status_log = self.log_path / f"{self.stage}.log"
        self.detail_log = self.log_path / "detail" / f"{self.stage}.detail.log"
        self.command_log = self.log_path / "commands" / f"{self.stage}.commands.log"
        self._status = None

    def is_timeout(self) -> bool:
        """Check if the log indicates a timeout."""
        try:
            with open(self.status_log, "r", encoding="utf-8") as f:
                content = f.read()
            return "subprocess.TimeoutExpired" in content
        except Exception:
            return False

    def get_command(self) -> str:
        """Get the command from the command log."""
        if self.command_log.exists():
            return self.command_log.read_text().strip()
        return ""

    @abstractmethod
    def get_error_details(self) -> tuple:
        """
        Extract error signature and error message from detail log.

        Returns:
            Tuple of (error_signature, error)
        """
        pass

    def get_status(self) -> str:
        """
        Determine the status.

        Returns:
            'timeout', 'error', or 'no_log'
        """
        if self.status_log.exists():
            return "timeout" if self.is_timeout() else "error"
        return "no_log"

    def to_dict(self) -> Dict:
        """
        Get all error information as a dictionary.

        Returns:
            Dictionary with status, command, error_signature, error
        """
        error_signature, error = self.get_error_details()
        return {
            f"{self.stage}_status": self.get_status(),
            "command": self.get_command(),
            "error_signature": error_signature,
            "error": error,
        }

    def populate_status_dict(self, status_dict: Dict[str, Dict], name: str):
        """
        Add error information to an existing status_dict entry.

        Args:
            status_dict: Dictionary to populate with error info
            name: The test name (key in status_dict)
        """
        if name not in status_dict:
            status_dict[name] = {}

        error_info = self.to_dict()
        for key, value in error_info.items():
            status_dict[name][key] = value


class CompilationErrorHandler(ErrorHandler):
    """Class to handle compilation error analysis and logging."""

    def get_error_details(self) -> tuple:
        """
        Extract error signature and error message from detail log.

        Returns:
            Tuple of (error_signature, error) where error_signature has
            operation names replaced with 'OP'
        """
        error_signature = ""
        error = ""
        try:
            with open(self.detail_log, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if "error:" in line:
                        error = line.split("error:")[-1].split(":")[0].strip()
                        error = error.split(";")[0]
                        error_signature = re.sub(r"'[^']*'", "'OP'", error)
                        return error_signature, error
        except Exception:
            pass
        return (error_signature, error)
