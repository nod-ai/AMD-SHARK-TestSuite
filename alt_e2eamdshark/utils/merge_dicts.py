import json
import argparse
from report import generate_report, save_dict
from pathlib import Path
import sys

TEST_DIR = str(Path(__file__).parents[1])
sys.path.append(TEST_DIR)
from run import ALL_STAGES


def _get_argparse():
    msg = "A script for loading two or more dictionary jsons and merging them."
    parser = argparse.ArgumentParser(prog="merge_dicts.py", description=msg, epilog="")
    parser.add_argument(
        "-s",
        "--sources",
        nargs="*",
        required=True,
        help="specify paths to source files for merging",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="specify output filepath",
    )
    parser.add_argument(
        "-r",
        "--report",
        action="store_true",
        default=False,
        help="set this flag to generate a report from the merged status dicts",
    )
    parser.add_argument(
        "-f",
        "--report-file",
        default="report.md",
        help="specify report filepath for merged status dicts",
    )
    return parser


def main(args):
    sources = args.sources
    print(sources)
    s0 = load(sources[0])
    for s in sources[1:]:
        s0.update(load(s))
    if args.output:
        save_dict(s0, args.output)
    if args.report_file:
        generate_report(args, ALL_STAGES, s0)


def load(pathname):
    with open(pathname) as contents:
        loaded_dict = json.load(contents)
    return loaded_dict


if __name__ == "__main__":
    parser = _get_argparse()
    main(parser.parse_args())
