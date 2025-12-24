#!/usr/bin/env python3
"""
Script to add issue links to the ERRORS table in a markdown report.

Reads a markdown file, finds the table in "## ERRORS (grouped by error)" section,
looks up each error in issue_list.csv, and adds a new Issue column with the link.

Usage:
    python add_issue_links.py -i <input.md> -m <issue_list.csv> -o <output.md>

Example:
    python add_issue_links.py -i test.md -m issue_list.csv -o output.md
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def load_issue_mapping(csv_file: str) -> dict:
    """Load error -> issue URL mapping from CSV file.

    Args:
        csv_file: Path to the issue_list.csv file

    Returns:
        Dictionary mapping error strings to issue URLs
    """
    mapping = {}
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            error = row.get("Error", "").strip()
            issue = row.get("Issue", "").strip()
            if error:
                mapping[error] = issue
    return mapping


def process_markdown(input_file: str, issue_mapping: dict, output_file: str):
    """Process markdown file and add issue links to the error table.

    Args:
        input_file: Path to input markdown file
        issue_mapping: Dictionary mapping error strings to issue URLs
        output_file: Path to output markdown file
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output_lines = []
    in_error_section = False
    in_error_table = False
    header_processed = False
    separator_processed = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if we're entering the ERRORS section
        if line.strip() == "## ERRORS (grouped by error)":
            in_error_section = True
            output_lines.append(line)
            i += 1
            continue

        # Check if we're leaving the ERRORS section (next ## header)
        if (
            in_error_section
            and line.strip().startswith("## ")
            and "ERRORS (grouped by error)" not in line
        ):
            in_error_section = False
            in_error_table = False
            header_processed = False
            separator_processed = False

        # Process table in the ERRORS section
        if in_error_section and line.strip().startswith("|"):
            parts = [p.strip() for p in line.strip().split("|")]
            # Remove empty strings from split
            parts = [p for p in parts if p or parts.index(p) in [0, len(parts) - 1]]

            if not header_processed:
                # This is the header row: | Error | Count |
                # Add Issue column: | Error | Count | Issue |
                new_line = "| Error | Count | Issue |\n"
                output_lines.append(new_line)
                header_processed = True
                in_error_table = True
                i += 1
                continue

            if header_processed and not separator_processed:
                # This is the separator row: |---|---|
                # Add separator for Issue column: |---|---|---|
                new_line = "|---|---|---|\n"
                output_lines.append(new_line)
                separator_processed = True
                i += 1
                continue

            if in_error_table and separator_processed:
                # This is a data row: | error_text | count |
                # Extract error and count
                match = re.match(r"\|\s*(.+?)\s*\|\s*(\d+)\s*\|", line.strip())
                if match:
                    error_text = match.group(1).strip()
                    count = match.group(2).strip()

                    # Look up issue URL
                    issue_url = issue_mapping.get(error_text, "")

                    # Format issue as link if URL exists
                    if issue_url:
                        # Extract issue number from URL for display
                        issue_num_match = re.search(r"/issues/(\d+)", issue_url)
                        if issue_num_match:
                            issue_link = f"[#{issue_num_match.group(1)}]({issue_url})"
                        else:
                            issue_link = f"[Link]({issue_url})"
                    else:
                        issue_link = ""

                    # Create new line with Issue column
                    new_line = f"| {error_text} | {count} | {issue_link} |\n"
                    output_lines.append(new_line)
                    i += 1
                    continue

        # Default: copy line as-is
        output_lines.append(line)
        i += 1

    # Write output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    print(f"Processed {input_file}")
    print(f"Output written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Add issue links to the ERRORS table in a markdown report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_issue_links.py -i test.md -m issue_list.csv -o output.md
  python add_issue_links.py --input-file report.md --issue-mapping issues.csv --output-file report_with_links.md
        """,
    )

    parser.add_argument(
        "-i", "--input-file", required=True, help="Input markdown file to process"
    )
    parser.add_argument(
        "-m",
        "--issue-mapping",
        required=True,
        help="CSV file with error -> issue URL mappings",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="Output markdown file with issue links added",
    )

    args = parser.parse_args()

    input_file = args.input_file
    csv_file = args.issue_mapping
    output_file = args.output_file

    # Verify files exist
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    if not Path(csv_file).exists():
        print(f"Error: CSV file not found: {csv_file}")
        sys.exit(1)

    # Load issue mapping
    print(f"Loading issue mappings from {csv_file}...")
    issue_mapping = load_issue_mapping(csv_file)
    print(f"Loaded {len(issue_mapping)} error -> issue mappings")

    # Process markdown
    print(f"Processing {input_file}...")
    process_markdown(input_file, issue_mapping, output_file)

    print("Done!")


if __name__ == "__main__":
    main()
