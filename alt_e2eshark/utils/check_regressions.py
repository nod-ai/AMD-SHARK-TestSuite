import json
import argparse
from pathlib import Path
from enum import Enum
import io
from typing import Dict


class StageOrder(Enum):
    setup = 0
    import_model = 1
    preprocessing = 2
    compilation = 3
    construct_inputs = 4
    native_inference = 5
    compiled_inference = 6
    postprocessing = 7
    results_summary = 8
    Numerics = 9
    PASS = 10


def _get_argparse():
    msg = "A script for loading two status dictionary jsons checking for regressions."
    parser = argparse.ArgumentParser(
        prog="check_regressions.py", description=msg, epilog=""
    )
    parser.add_argument(
        "--old",
        required=True,
        help="specify path to old status dict json",
    )
    parser.add_argument(
        "--new",
        required=True,
        help="specify path to new status dict json",
    )
    parser.add_argument(
        "-f",
        "--report-file",
        default="regression_report.md",
        help="specify filepath for regression report",
    )
    parser.add_argument(
        "-r",
        "--perf_tol_regression",
        default="0.05",
        help="specify a minimum percent difference required to report a perf regression",
    )
    parser.add_argument(
        "-p",
        "--perf_tol_progression",
        default="0.05",
        help="specify a minimum percent difference required to report a perf progression",
    )
    return parser


def save_dict(status_dict: Dict[str, Dict], status_dict_json: str):
    with io.open(status_dict_json, "w", encoding="utf8") as outfile:
        dict_str = json.dumps(
            status_dict,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
            ensure_ascii=False,
        )
        outfile.write(dict_str)


def get_table_header(ordered_items):
    s0 = "|model name|"
    s1 = "|---|"
    for item in ordered_items:
        s0 += f"{item}|"
        s1 += "---|"
    s = s0 + "\n" + s1 + "\n"
    return s


def get_table_rows(some_dict: Dict, ordered_items) -> str:
    s = ""
    for key, d in some_dict.items():
        s += f"|{key}|"
        for item in ordered_items:
            if item not in d.keys():
                s += " |"
                continue
            s += f"{d[item]}|"
        s += "\n"
    return s


def get_comp_string(prog_regr_dict: Dict, kind_of_change: str = "Regression") -> str:
    s = "## "
    num_prog = len(prog_regr_dict.keys())
    if num_prog == 0:
        return s + f"No {kind_of_change}s Found\n\n"
    elif num_prog == 1:
        s += f"One {kind_of_change} Found:\n\n"
    else:
        s += f"{num_prog} {kind_of_change}s Found:\n\n"

    ordered_items = ["old_status", "new_status"]
    s += get_table_header(ordered_items)
    s += get_table_rows(prog_regr_dict, ordered_items)
    return s + "\n"


def get_perf_string(perf_tol: Dict, perf_comp: Dict):
    if len(perf_comp.keys()) == 0:
        return ""
    s = "## Performance Comparison\n\n"
    s += f"regression tolerance: {round(100*perf_tol['perf_tol_regression'],1)}%\n\n"
    s += f"progression tolerance: {round(100*perf_tol['perf_tol_progression'],1)}%\n\n"

    ordered_items = [
        "exit_status",
        "analysis",
        "old_time_ms",
        "new_time_ms",
        "change_ms",
        "percent_change",
    ]
    s += get_table_header(ordered_items)
    s += get_table_rows(perf_comp, ordered_items)
    return s + "\n"


def save_comp_report(comp_dict: Dict, filepath):
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    perf_string = get_perf_string(comp_dict["perf_tolerances"], comp_dict["perf_comp"])
    regr_string = get_comp_string(comp_dict["regressions"], "Regression")
    prog_string = get_comp_string(comp_dict["progressions"], "Progression")
    s = "# Test Run Comparison Report\n\n"
    s += perf_string
    s += regr_string
    s += prog_string
    with open(filepath, "w") as file:
        file.write(s)


def check_regressions(
    new: Dict, old: Dict, tol_regr: float = 0.05, tol_prog: float = 0.05
) -> Dict:
    regressions = dict()
    progressions = dict()
    perf_comp = dict()
    for key, d in new.items():
        if key not in old.keys():
            continue

        old_d = old[key]

        new_status = d["exit_status"]
        old_status = old_d["exit_status"]

        # status comparison:
        new_order = StageOrder[new_status].value
        old_order = StageOrder[old_status].value
        if new_order < old_order:
            regressions[key] = {"old_status": old_status, "new_status": new_status}
        if new_order > old_order:
            progressions[key] = {"old_status": old_status, "new_status": new_status}

        # perf comparison
        new_time_ms = d["time_ms"]
        old_time_ms = old_d["time_ms"]

        if isinstance(new_time_ms, float) and isinstance(old_time_ms, float):
            delta = new_time_ms - old_time_ms
            if old_time_ms <= 0.0:
                continue
            delta_percent = delta / old_time_ms
            perf_comp[key] = {
                "exit_status": new_status,
                "old_time_ms": round(old_time_ms, 4),
                "new_time_ms": round(new_time_ms, 4),
                "change_ms": round(delta, 4),
                "percent_change": f"{round(100*delta_percent,2)}%",
            }
            if delta_percent > abs(tol_regr):
                perf_comp[key]["analysis"] = "regression"
            elif delta_percent < -abs(tol_prog):
                perf_comp[key]["analysis"] = "progression"
            else:
                perf_comp[key]["analysis"] = "within tol"
    perf_tol = {
        "perf_tol_regression": abs(tol_regr),
        "perf_tol_progression": abs(tol_prog),
    }
    combined_dict = {
        "perf_tolerances": perf_tol,
        "regressions": regressions,
        "progressions": progressions,
        "perf_comp": perf_comp,
    }
    return combined_dict


def main(args):
    new = load(args.new)
    old = load(args.old)
    combined_dict = check_regressions(
        new, old, float(args.perf_tol_regression), float(args.perf_tol_progression)
    )
    save_comp_report(combined_dict, args.report_file)
    report_path = Path(args.report_file)
    parent_path = report_path.parent
    file_name = report_path.stem
    json_path = parent_path / f"{file_name}.json"
    save_dict(combined_dict, json_path)


def load(pathname):
    with open(pathname) as contents:
        loaded_dict = json.load(contents)
    return loaded_dict


if __name__ == "__main__":
    parser = _get_argparse()
    main(parser.parse_args())
