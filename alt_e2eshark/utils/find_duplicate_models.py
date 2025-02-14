from pathlib import Path
import argparse
from typing import Union, Dict, Any, Optional
import json
import io

ROOT = Path(__file__).parents[1]


class HashableDict(dict):
    """a hashable dictionary, used to invert a dictionary with dictionary values"""

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def load_json_dict(filepath: Union[str, Path]) -> Dict[str, Any]:
    with open(filepath) as contents:
        loaded_dict = json.load(contents)
    return loaded_dict


def save_to_json(jsonable_object, name_json: Optional[str] = None):
    """Saves an object to a json file with the given name, or prints result."""
    dict_str = json.dumps(
        jsonable_object,
        indent=4,
        sort_keys=True,
        separators=(",", ": "),
        ensure_ascii=False,
    )
    if not name_json:
        print(dict_str)
        return
    path_json = ROOT / f"{Path(name_json).stem}.json"
    with io.open(path_json, "w", encoding="utf8") as outfile:
        outfile.write(dict_str)


def get_groupings(metadata_dicts: Dict[str, Dict]) -> Dict:
    """gets a multi-valued inverse of metatdata_dicts"""
    groupings = dict()
    for key, value in metadata_dicts.items():
        value["op_frequency"] = HashableDict(value["op_frequency"])
        hashable = HashableDict(value)
        if hashable in groupings.keys():
            groupings[hashable].append(key)
        else:
            groupings[hashable] = [key]
    return groupings


def main(args):
    run_dir = ROOT / args.rundirectory
    metadata_dicts = dict()
    for x in run_dir.glob("**/metadata.json"):
        test_name = x.parent.name
        metadata_dicts[test_name] = load_json_dict(x)

    groupings = get_groupings(metadata_dicts)
    found_redundancies = []
    for key, value in groupings.items():
        if len(value) > 1:
            found_redundancies.append(
                value if args.simplified else {"models": value, "shared_metadata": key}
            )
    save_to_json(found_redundancies, args.output)


def _get_argparse():
    msg = "After running run.py with the flag --get-metadata, use this tool to find duplicate models."
    parser = argparse.ArgumentParser(
        prog="find_duplicate_models.py", description=msg, epilog=""
    )

    parser.add_argument(
        "-r",
        "--rundirectory",
        default="test-run",
        help="The directory containing run.py results",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="specify an output json file",
    )
    parser.add_argument(
        "-s",
        "--simplified",
        action="store_true",
        default=False,
        help="pass this arg to only print redundant model lists, without the corresponding metadata.",
    )
    return parser


if __name__ == "__main__":
    parser = _get_argparse()
    main(parser.parse_args())
