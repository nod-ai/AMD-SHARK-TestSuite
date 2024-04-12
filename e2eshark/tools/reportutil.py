# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os, sys, argparse, tabulate, pickle


def loadTable(reportpkl):
    table = None
    if os.path.exists(reportpkl):
        with open(reportpkl, "rb") as pkf:
            table = pickle.load(pkf)
        pkf.close()
    return table


def checkAndGetHeader(headers, column_indices):
    retheader = []
    # Check if headers match, else cannot process
    for i, item in enumerate(headers):
        for j in range(i + 1, len(headers)):
            if headers[i] != headers[j]:
                print(f"The headers of reports in given run dirs differ.")
                print(
                    f"One of them had: {headers[i]}\nWhile the other had:{headers[j]}"
                )
                sys.exit(1)
    # Check column indices
    for i in column_indices:
        if i < 0 or i >= len(headers[0]):
            print(
                f"The --column index {i} is out of range. Valid range is: {0}:{len(headers[0])-1}"
            )
            sys.exit(1)

    # pick one of the headers as merged header to be build
    if len(headers) > 0:
        retheader = selectColumns(headers[0], column_indices)
    return retheader


def selectColumns(row, column_indices):
    # Assumes indices have been chedked in column_indices within range earlier
    if len(column_indices) == 0:
        return row
    selectedcolumns = [row[i] for i in column_indices]
    return selectedcolumns


def createMergedHeader(args, runnames, header):
    if args.reportformat == "pipe" or args.reportformat == "html":
        joiner = "\n"
    else:
        joiner = "."
    if args.mode == "summary":
        mergedheader = ["items"]
    else:
        mergedheader = ["tests"]
    for i in range(len(header)):
        for run in runnames:
            columnname = header[i] + joiner + run
            mergedheader += [columnname]
    return mergedheader


def getCanonicalizedListOfRuns(args, runnames, dictOfRuns, column_indices, rowlen):
    listOfRuns = []
    for run in runnames:
        if dictOfRuns.get(run):
            listOfRuns += [selectColumns(dictOfRuns[run], column_indices)]
        else:
            listOfRuns += [
                ["NA" if args.mode == "status" else 0 for i in range(rowlen)]
            ]
    return listOfRuns


def createOneMergedRow(
    args, runnames, firstColumnIdentifier, dictOfRuns, column_indices, rowlen
):
    merged = [firstColumnIdentifier]
    listOfRuns = getCanonicalizedListOfRuns(
        args, runnames, dictOfRuns, column_indices, rowlen
    )
    # zip creates a tuple by taking same index value from each of the
    # unpacked (using * operator) run in listOfRuns
    for cellItemTuple in zip(*listOfRuns):
        merged.extend(cellItemTuple)
    return merged


def createMergedRows(args, runnames, reportdict, column_indices, rowlen):
    mergedrows = []
    for test, dictOfRuns in reportdict.items():
        merged = createOneMergedRow(
            args, runnames, test, dictOfRuns, column_indices, rowlen
        )
        mergedrows += [merged]

    return mergedrows


def getDiff(args, tuple, diff):
    # if it is a two element tuple, then provide exact difference
    # for the pair for numbers, else say differ of match
    if len(tuple) == 2:
        if args.mode == "time" or args.mode == "summary":
            tuple = [float(i) if isinstance(i, str) else i for i in tuple]
            if isinstance(tuple[0], int):
                elemdiffnum = int(tuple[1]) - int(tuple[0])
                elemdiff = str(elemdiffnum)
                if args.verbose:
                    if elemdiffnum == 0:
                        elemdiff += " (same)"
                    elif elemdiffnum > 0:
                        elemdiff += " (improved)"
                    else:
                        elemdiff += " (regressed)"
            else:
                elemdiffnum = float(tuple[1]) - float(tuple[0])
                elemdiff = f"{elemdiffnum:.{3}f}"

            diff.extend([elemdiff])
            return

    # Else do exact match for all
    matched = all(tuple[0] == j for j in tuple)
    if matched:
        diff.extend(["same"])
    else:
        diffidentifier = "differ"
        if args.verbose:
            if args.mode == "time" or args.mode == "summary":
                if isinstance(tuple[0], float):
                    tuple = [f"{i:.{3}f}" for i in tuple]
                elif isinstance(tuple[0], int):
                    tuple = [str(i) for i in tuple]
                else:
                    tuple = [f"{j:.{3}f}" for j in [float(i) for i in tuple]]

            diffidentifier = "[" + ",".join(tuple) + "]"
        diff.extend([diffidentifier])
    return diff


def createDiffRows(args, runnames, reportdict, column_indices, rowlen):
    diffrows = []
    for test, dictOfRuns in reportdict.items():
        diff = [test]
        listOfRuns = getCanonicalizedListOfRuns(
            args, runnames, dictOfRuns, column_indices, rowlen
        )
        # zip creates a tuple by taking same index value from each of the
        # unpacked (using * operator) run in listOfRuns
        for cellItemTuple in zip(*listOfRuns):
            getDiff(args, cellItemTuple, diff)
        diffrows += [diff]
    return diffrows


def convertNumToString(rows):
    strrows = []
    for row in rows:
        strrows += [[str(i) for i in row]]
    return strrows


def convertStringToFloat(rows):
    floatrow = []
    for row in rows:
        floatrow += [[float(i) for i in row]]
    return floatrow


def addTestsToDict(
    reportdict, reportpkl, runname, skiporincludetestslist, skiporinclude
):
    table = loadTable(reportpkl)
    # skip test name, hence from 1
    header = [table[0][1:]]
    # skip table header, hence index from 1
    for i in range(1, len(table)):
        testname = table[i][0]
        if skiporinclude == "skip" and testname in skiporincludetestslist:
            continue
        elif skiporinclude == "include" and not testname in skiporincludetestslist:
            continue
        # Add to dictionary of testname to dictionary of run name
        if reportdict.get(testname):
            reportdict[testname][runname] = table[i][1:]
        else:
            reportdict[testname] = {runname: table[i][1:]}

    return header


def createMergedReport(args, reportdict, runnames, header, column_indices):
    # Create merged header
    mergedheader = createMergedHeader(args, runnames, header)
    mergedrows = createMergedRows(
        args, runnames, reportdict, column_indices, len(header)
    )
    mergedtable = tabulate.tabulate(
        mergedrows, headers=mergedheader, tablefmt=args.reportformat
    )
    return mergedtable


def createDiffReport(args, reportdict, runnames, header, column_indices):
    if args.mode == "summary":
        diffheader = ["items"] + header
    else:
        diffheader = ["test-name"] + header
    diffrows = createDiffRows(args, runnames, reportdict, column_indices, len(header))
    difftable = tabulate.tabulate(
        diffrows, headers=diffheader, tablefmt=args.reportformat
    )
    return difftable


def getTestsListFromFile(testlistfile):
    testlist = []
    if not os.path.exists(testlistfile):
        print(f"The file {testlistfile} does not exist")
    with open(testlistfile, "r") as tf:
        testlist += tf.read().splitlines()
    testlist = [item.strip().strip(os.sep) for item in testlist]
    return testlist


if __name__ == "__main__":
    msg = "The script to diff and combine reports generated by e2eshark run.pl"
    parser = argparse.ArgumentParser(description=msg, epilog="")
    parser.add_argument(
        "inputdirs",
        nargs="*",
        help="Input test run directory names",
    )
    parser.add_argument(
        "-c",
        "--columns",
        help="Provide a a,b,c indices to select header columns to inculde in report. Default is all.",
    )
    parser.add_argument(
        "-d",
        "--do",
        choices=["diff", "merge"],
        default="merge",
        help="Merge the reports in two directories to create or diff the two reports",
    )
    parser.add_argument(
        "-f",
        "--reportformat",
        choices=["pipe", "github", "html", "csv"],
        default="pipe",
        help="Format of the test report summary file. It takes subset of tablefmt value of python tabulate",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write merged outout into this file. Default is to display on stdout.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["status", "time", "summary"],
        default="status",
        help="Process status report, time report, or summary report (count of passes)",
    )
    parser.add_argument(
        "-s",
        "--skiptestsfile",
        help="A file with lists of tests that should be skipped from consideration",
    )
    parser.add_argument(
        "-t",
        "--testsfile",
        help="A file with lists of only tests (if present in the run report) to consider",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print details in the reports",
    )

    args = parser.parse_args()
    if args.skiptestsfile and args.testsfile:
        print(f"Only one of --skiptestsfile or --testsfile can be used")
        sys.exit(1)

    dirlist = args.inputdirs
    reportdict = {}
    allheaders = []
    runnames = []
    testlist = []
    mergedreportfilename = ""
    column_indices = []
    skiporincludetestslist = []
    skiporinclude = None
    # Filter the columns using the column indices
    if args.columns:
        indices_string_list = args.columns.split(",")
        column_indices = [int(str) for str in indices_string_list]
    if args.skiptestsfile:
        skiporincludetestslist = getTestsListFromFile(args.skiptestsfile)
        skiporinclude = "skip"
    if args.testsfile:
        skiporincludetestslist = getTestsListFromFile(args.testsfile)
        skiporinclude = "include"

    for item in dirlist:
        rundir = os.path.abspath(item)
        runname = os.path.basename(rundir)
        runnames += [runname]
        if not os.path.exists(rundir):
            print("The given file ", rundir, " does not exist\n")
            sys.exit(1)
        if args.mode == "time":
            reportpkl = rundir + "/timereport.pkl"
        elif args.mode == "status":
            reportpkl = rundir + "/statusreport.pkl"
        elif args.mode == "summary":
            reportpkl = rundir + "/summaryreport.pkl"
        else:
            print(f"The mode {args.mode} is not supported")
            sys.exit(1)

        if not os.path.exists(reportpkl):
            print(f"{reportpkl} does not exist. This report will be ignored.")
            continue

        allheaders += addTestsToDict(
            reportdict,
            reportpkl,
            runname,
            skiporincludetestslist,
            skiporinclude,
        )
    if len(allheaders) == 0:
        print(f"No valid reports found")
        sys.exit(1)
    oneheader = checkAndGetHeader(allheaders, column_indices)
    if args.do == "merge":
        outtable = createMergedReport(
            args, reportdict, runnames, oneheader, column_indices
        )
    elif args.do == "diff":
        outtable = createDiffReport(
            args, reportdict, runnames, oneheader, column_indices
        )

    outf = sys.stdout
    if args.output:
        outf = open(args.output, "w")
    runstr = ", ".join(runnames)
    extramsg = ""
    if args.mode == "time":
        extramsg = "(in seconds)"
    elif args.mode == "summary":
        extramsg = "(time in seconds)"

    print(
        f"The {args.do} report for {args.mode} {extramsg} for runs: {runstr}", file=outf
    )
    print(outtable, file=outf)
    sys.exit(0)
