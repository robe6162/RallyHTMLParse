#!/usr/bin/env python

import argparse
import csv
import json
import math
import logging
import operator
import pprint
import os
import re
import sys

import prettytable

# In python 2, reduce is a native call, but in Python3, it has been incorporated
# into functools.
PY_VERSION = sys.version_info.major
if PY_VERSION > 2:
    from functools import reduce


DEFAULT_LOG_FILE = "rally_html_parse.log"


class TableConfig(object):
    CSV = 'csv'
    PRETTY = 'prettytable'
    WIKI = 'wiki'
    JSON = 'json'
    FORMAT = '{0:0.3f}'
    FORMAT_INT = '{0:d}'
    TYPES = [PRETTY, WIKI, CSV, JSON]


class RallyReportConsts(object):
    FULL_DUR = 'full_duration'
    LOAD_DUR = 'load_duration'
    TABLE = 'table'
    CLASS = 'cls'
    NAME = 'name'
    COLS = 'cols'
    ROWS = 'rows'
    SUCCESS = 'Success'
    COUNT = 'Count'
    ACTION = 'Action'
    MIN = 'Min'
    MAX = 'Max'
    MEDIAN = 'Median'
    AVG = 'Avg'
    NINETY_FIFTH = '95%ile'
    NINETYTH = '90%ile'
    SUMMARY_EXCLUDE = [ACTION]
    DETAIL_EXCLUDE = []
    TABLE_ORDER = [
        COUNT, MIN, MEDIAN, MAX, AVG, NINETYTH, NINETY_FIFTH, SUCCESS]
    TEST = 'Test'
    TOTAL = 'total'
    RUN_ID = 'RUN ID'


class ParseRallyResults(object):

    DATA_GROUP_NAME = 'data'
    REGEX_SCENARIO = \
        r'\$scope\.scenarios\s=\s(?P<data>.*)\s*;\s*\$scope.location'

    def __init__(self, rally_html_file, id_, summary_only=False, debug=False,
                 target=None):

        self._src = rally_html_file
        self.id_ = id_
        self.raw_data = None
        self.results = None
        self.results_tally = None
        self.results_summary = None
        self.debug = debug
        self.target = target

        self.summary_only = summary_only

        self.parse(summary_only=self.is_summary_only())

    def is_summary_only(self):
        """
        Get summary only flag

        @return: boolean
        """
        return self.summary_only

    def parse(self, filename=None, summary_only=False):
        filename = filename or self._src
        self.raw_data = self.read_file(filename=filename)
        self.results = self.get_results_data()
        self.results_tally = self.tally_results(summary_only=summary_only)
        self.results_summary = self.summarize_results()

    @staticmethod
    def read_file(filename):
        """
        Read the html file into memory.

        @return: Lines appended into a single string
        """
        log.info("Parsing: '{0}'".format(os.path.abspath(filename)))
        with open(filename, "r") as DATA:
            file_data = "".join(DATA.readlines())
        return file_data

    def get_results_data(self):
        """
        Find and return JSON "data" portion of Javascript within HTML source.

        @return: data structure created from JSON output.
        """

        # Find match in HTML source
        match = re.search(self.REGEX_SCENARIO, self.raw_data, re.IGNORECASE)
        results_data = []

        # If match is found, convert match to data structure from JSON
        if match is not None:
            results_data = json.loads(match.group(self.DATA_GROUP_NAME))

        return results_data

    def tally_results(self, summary_only=False):
        """
        Tally up total timings from each set of tests. Excluding fields defined
        in RallyReportsConsts.EXCLUDE

        @return: Dictionary of lists
            Key1 = Test Name
            Key2 = Subaction Name
            Key3 = Metric (avg, mean, percentile)
            Value: List of values for metric
        """
        results = dict()

        for datum in self.results:

            name = str(datum[RallyReportConsts.NAME].split(' ')[0])

            # Get the table element from dictionary
            table = datum[RallyReportConsts.TABLE]

            # Define the subactions
            subactions = [str(x[0].split(' ')[0]) for x in table[
                RallyReportConsts.ROWS]]
            metrics = [str(x.split(' ')[0]) for x in
                       table[RallyReportConsts.COLS]]

            if self.debug and name == self.target:
                log.debug("NAME: {0}".format(name))
                log.debug("TABLE: {0}".format(table))
                log.debug("ACTIONS: {0}".format(subactions))
                log.debug("METRICS: {0}".format(metrics))

            # If the test is not in the dictionary, create the key, and the
            # value a dictionary of lists for each metric (if not EXCLUDED).
            if name not in results:
                results[name] = dict()

            for action in subactions:

                # If only a summary should be generated, ignore all metrics
                # except total
                if summary_only:
                    if action != RallyReportConsts.TOTAL:
                        continue

                # Build out the results structure
                if action not in results[name].keys():
                    results[name][str(action)] = \
                        dict([(str(key.split(' ')[0]), list()) for key in
                              metrics if key not
                              in RallyReportConsts.SUMMARY_EXCLUDE])

            # Append each value to the corresponding metric list, based on
            # the test name.
            action_counter = 0
            for action in subactions:

                # If only a summary is being generated, ignore all actions
                # except total.
                if summary_only:
                    if action != RallyReportConsts.TOTAL:
                        continue

                # Get the data for the current test action
                data = table[RallyReportConsts.ROWS][action_counter]
                keyed_data = dict(zip(metrics, data))

                if self.debug and name == self.target:
                    log.debug("NAME: {0}.{1}".format(name, action))
                    log.debug("METRICS: {0}".format(metrics))
                    log.debug("TABLE: {0}".format(table[RallyReportConsts.ROWS]))
                    log.debug("DATA: {0}".format(keyed_data))

                for metric in metrics:
                    if metric in results[name][action]:
                        results[name][action][metric].append(
                            keyed_data[metric])

                action_counter += 1

        results = self._scrub_success_values(results)

        if self.debug:
            log.debug("RESULTS")
            log.debug("=" * 60)
            log.debug(pprint.pformat(results))

        return results

    @staticmethod
    def _scrub_success_values(results_dict):
        """
        The success rate is a unicode string with the format u'yyy.x%'.
        Cast to a string, strip off the '%, and if it is 'n/a', set it to 0

        :param results_dict: results dictionary (same format generated by:
        ParseRallyResults.tally_results()

        :return: Updated dictionary

        """
        log_fmt_str = "{type_}: {test} {action} {stat}: {value}"
        for test in results_dict.keys():
            for action in results_dict[test].keys():
                for stat in results_dict[test][action].keys():
                    if stat == RallyReportConsts.SUCCESS:
                        success_values = results_dict[test][action][stat]
                        log.debug(log_fmt_str.format(
                            type_="Original", test=test, action=action,
                            stat=stat, value=success_values))

                        updated_values = [
                            str(x) if 'n/a' not in x.lower() else 0 for x in
                            success_values]

                        value = [int(float(str(val).strip('%'))) for val in
                                 updated_values]

                        results_dict[test][action][stat] = value
                        log.debug(log_fmt_str.format(
                            test=test, action=action, stat=stat, value=value,
                            type_="Processed"))

        return results_dict

    def summarize_results(self, summary_only=False):
        """
        Reduce the total results tally into a single value per metric for
        each test.

        Summary_only = Only summarize the test total, and do not include the
        test subactions.

        Averages and percentiles are averages of the entire population
        Min is the minimum for all min values
        Max is the maximum for all max values

        @return: Dictionary of dictionaries
        Key1 = Test
        Key2 = Metric
        Value = Summary of metric values overall
        """
        summary = dict()
        table = self.results_tally
        if table is None:
            table = self.tally_results(summary_only=summary_only)

        # For each test...
        for name in table.keys():
            summary[name] = dict()

            # For each metric in the test...
            for action in table[name].keys():

                log.debug("NAME: {name}   ACTION: {action}".format(
                    action=action, name=name))
                log.debug("METRICS: {0}".format(", ".join(
                    table[name][action].keys())))

                if action not in summary[name]:
                    summary[name][str(action)] = dict()

                for metric in table[name][action].keys():

                    # Strip out unicode (how is dependent on Python Version)
                    # In V3, 'unicode' has been rolled up into 'str'
                    if PY_VERSION == 2:
                        value_list = [x for x in table[name][action][metric] if
                                      not isinstance(x, unicode)]
                    else:
                        value_list = [x for x in table[name][action][metric] if
                                      not isinstance(x, str)]

                    log.debug(
                        "LIST OF KNOWN VALUES for METRIC '{0}': {1}".format(
                            metric, value_list))

                    if self.debug and name == self.target:
                        log.debug("NAME: {0}".format(name))
                        log.debug("ACTION: {0}".format(action))
                        log.debug("METRIC: {0}".format(metric))
                        log.debug("VALUES: {0}".format(value_list))

                    # Determine the MAX
                    if value_list:

                        # Determine the MAX
                        if metric == RallyReportConsts.MAX:
                            summary[name][action][metric] = max(value_list)

                        # Determine the MIN
                        elif metric == RallyReportConsts.MIN:
                            summary[name][action][metric] = min(value_list)

                        # Determine the MEDIAN
                        elif metric == RallyReportConsts.MEDIAN:
                            medians = sorted(value_list)
                            length = len(medians)
                            bisect = int(length/2) if length % 2 == 0 else int(
                                math.floor(length/2))
                            summary[name][action][metric] = medians[bisect]

                        elif metric == RallyReportConsts.COUNT:
                            summary[name][action][metric] = \
                                int(sum(value_list))

                        # Determine the average percentiles and averages
                        else:
                            numerator = reduce(
                                lambda v1, v2: v1 + v2, value_list)
                            summary[name][action][metric] = numerator / len(
                                value_list)
                    else:
                        summary[name][action][metric] = 'n/a'

                    if self.debug and name == self.target:
                        log.debug("VALUE: {0}".format(
                            summary[name][action][metric]))

                if self.debug and name == self.target:
                    log.debug("ACTION: {0}".format(action))
                    log.debug(pprint.pformat(summary[self.target]))

        if self.debug:
            log.debug("SUMMARY")
            log.debug("=" * 60)
            log.debug(pprint.pformat(summary))

        return summary


class RallYReportOutput(object):
    """
    Used to create the various output formats
    """

    def __init__(self, data):
        self.data = data

    def build_table(self, table_type=TableConfig.CSV, filename=None,
                    append=False, summary_only=False):
        """
        Build table of results for output.
        table_type: TableTypes.CSV, TableTypes.PRETTY, TableType.WIKI
        filename: Name of file to output table. (DEFAULT: None)

        @return: String output of resulting table
        """

        table = ''

        # Set up list of column names
        # Columns: ID, Test Name, [TABLE ORDER]
        columns = [RallyReportConsts.RUN_ID, RallyReportConsts.TEST,
                   RallyReportConsts.ACTION]
        columns.extend(RallyReportConsts.TABLE_ORDER)

        # Determine file access permissions (for writing to file)
        file_perms = 'a+' if append else 'wb'
        file_exists = os.path.exists(filename) if filename is not None \
            else False

        verb = "Wrote" if not append or not file_exists else "Appended"

        # If using PrettyTable (output to STDOUT)
        if table_type.lower() == TableConfig.PRETTY.lower():
            table = self.write_prettytable_results(
                filename=filename, file_perms=file_perms, columns=columns,
                verb=verb)

        # Writing to CSV
        elif table_type.lower() == TableConfig.CSV.lower():
            table = self.write_csv_results_file(
                filename=filename, file_perms=file_perms, append=append,
                file_exists=file_exists, columns=columns, verb=verb)

        elif table_type.lower() == TableConfig.WIKI.lower():
            table = self.write_wiki_results_file(
                filename=filename, file_perms=file_perms, append=append,
                summary_only=summary_only, columns=columns, verb=verb)

        elif table_type.lower() == TableConfig.JSON.lower():
            table = self.write_json_results_file(
                filename=filename, file_perms=file_perms, append=append,
                verb=verb)

        else:
            log.error("ERROR: Output format not recognized: {0}".format(
                table_type))

        return table

    # -------------------------------------------------------------------------
    #  FILE OUTPUT ROUTINES
    # -------------------------------------------------------------------------
    def write_prettytable_results(self, filename, file_perms, columns, verb):
        """
        Write the table results via PrettyTable
        :param filename: Name of file to output table
        :param file_perms: append or create
        :param columns: list of columns to include in the table
        :param verb: Past tense of action: 'appended' or 'wrote'

        :return: String representation of the table

        """
        default_alignment = 'r'
        cols_align_left = [RallyReportConsts.TEST, RallyReportConsts.ACTION]

        table = prettytable.PrettyTable()
        table.field_names = columns

        # Setup table column specifics (Sort by + write justifications)
        table.sortby = RallyReportConsts.TEST
        for col in columns:
            table.align[col] = (
                'l' if col in cols_align_left else default_alignment)

        # Iterate through the tests
        data = self.data.results_summary
        for test_name, data in data.iteritems():

            # Populate the row with data from the corresponding header
            for action, stats in data.iteritems():
                data_row = [self.data.id_, test_name, action]

                for datum in RallyReportConsts.TABLE_ORDER:
                    value = stats[datum]

                    # Try to format values, if not a string
                    try:
                        value = TableConfig.FORMAT.format(value)
                        if datum == RallyReportConsts.COUNT:
                            value = TableConfig.FORMAT_INT.format(value)
                    except ValueError:
                        pass
                    data_row.append(value)

                table.add_row(data_row)

        # If filename is set, write to file
        if filename is not None:
            table_text = table.get_string()

            with open(filename, file_perms) as OUT:
                OUT.write(table_text)
                OUT.write('\n')

            log.info("{verb} table to {out}".format(
                out=os.path.abspath(filename), verb=verb))

        table = table.get_string(sort_key=operator.itemgetter(2, 3),
                                 sortby=RallyReportConsts.TEST)
        return table

    def write_csv_results_file(
            self, filename, file_perms, append, file_exists, columns, verb):
        """
        Write the results output in a CSV format

        :param filename: Name of file to write CSV
        :param file_perms: Append ('a') or write ('w')
        :param append: (Boolean) Should we append, if so, don't create the
                           headers
        :param file_exists: Should the file exist?
        :param columns: List of columns to report
        :param verb: Past tense of action: 'appended' or 'wrote'

        :return: String representation of the CSV output

        """
        with open(filename, file_perms) as CSV_FILE:
            writer = csv.writer(CSV_FILE, quotechar='|')

            table = ''
            # Only add the header if the file DNE or appending to
            # existing file
            if not append or not file_exists:
                writer.writerow(columns)
            table += "{0}\n".format(columns)

            # Write data to CSV file
            data = self.data.results_summary()
            for test_name in sorted(data.keys()):
                for action in sorted(data[test_name].keys()):
                    stats = data[test_name][action]

                    data_row = [self.data.id_(), test_name, action]
                    for datum in RallyReportConsts.TABLE_ORDER:
                        value = stats[datum]

                        # Try to format values, if not a string
                        try:
                            value = TableConfig.FORMAT.format(value)
                            if datum == RallyReportConsts.COUNT:
                                value = TableConfig.FORMAT_INT.format(value)
                        except ValueError:
                            pass
                        data_row.append(value)
                        table += '{0}\n'.format(data_row)

                    writer.writerow(data_row)

        log.info("{verb} data to {out}.".format(verb=verb, out=filename))
        return table

    def write_wiki_results_file(
            self, filename, file_perms, append, summary_only, columns, verb):
        """
        Write the results in a wiki shorthand format

        :param filename: Name of the file to write output
        :param file_perms: "w", "a", "w+"
        :param append: (Boolean) Should we append, if so, don't create the
                           headers
        :param summary_only: Only output the summary, not the details
        :param columns: List of columns to report
        :param verb: Past tense of action: 'appended' or 'wrote'

        :return: String representation of the generated wiki shorthand

        """
        data_sets = [self.data.results_summary]

        if not append and filename is not None and os.path.exists(filename):
            os.remove(filename)

        if not summary_only:
            file_perms = 'a+'
            new_data_set = []

            # Data set [0] is used because when the data is passed into
            # the routine, it is a single element test
            for test_name in sorted(data_sets[0].keys()):
                test_dict = dict([(test_name, dict())])
                for action in sorted(data_sets[0][test_name].keys()):
                    test_dict[test_name][action] = data_sets[0][test_name][
                        action]
                    if action == RallyReportConsts.TOTAL:
                        break

                new_data_set.append(test_dict)
            data_sets = new_data_set

        table = '\n'
        for datum in data_sets:
            table += '{0}\n'.format(self.build_wiki_format(
                columns=columns, data=datum, filename=filename,
                file_perms=file_perms))

        log.info("{verb} data to {out}.".format(verb=verb, out=filename))
        return table

    def build_wiki_format(
            self, columns, data, filename=None, file_perms='w'):
        """
        Build the actual wiki shorthand notation for the table

        :param columns: List of columns to display
        :param data: Data for the table
        :param filename: Name of file to output
        :param file_perms: File permissions

        :return: String representation of the generated wiki shorthand
        """

        # HEADER String Definition
        table = '||{table}||\n'.format(table='||'.join(columns))
        for test_name in sorted(data.keys()):
            for action in sorted(data[test_name].keys()):
                stats = data[test_name][action]

                data_row = [self.data.id_, test_name, action]
                for datum in RallyReportConsts.TABLE_ORDER:
                    value = stats[datum]

                    # Try to format values, if not a string
                    try:
                        value = TableConfig.FORMAT.format(value)
                        if datum == RallyReportConsts.COUNT:
                            value = TableConfig.FORMAT_INT.format(value)
                    except ValueError:
                        pass
                    data_row.append(value)

                # ROW String Definition (based on header columns)
                table_row = "|{row}|\n".format(row='|'.join(data_row))
                table += table_row

            if filename is not None:
                with open(filename, file_perms) as WIKI_FILE:
                    for line in table.split('\n'):
                        WIKI_FILE.write('{0}\n'.format(line))

        return table

    def write_json_results_file(self, filename, file_perms, append, verb):
        """
        Write results to file in JSON format

        :param filename: Name of file to write JSON
        :param file_perms: "w", "a", "w+"
        :param append: Boolean - Should the file be appended to?
        :param verb: Past tense of action: "Appended", "Wrote"

        :return: String representation of JSON formatted data
        """

        if not append and filename is not None and os.path.exists(filename):
            os.remove(filename)

        # Write data to JSON file
        json_data = json.dumps(self.data.results_summary)
        with open(filename, file_perms) as JSON_FILE:
            JSON_FILE.write(json_data)

        log.info("{verb} data to {out}.".format(verb=verb, out=filename))
        return json_data


def parse_cli_args():
    """
    Define command line arguments
    @return: ArgParser argument object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Name of input Rally Results HTML file.')
    parser.add_argument('id', help='Identifier for test data. e.g. - run id '
                                   'or date.')

    # -------------------------------

    formatting = parser.add_argument_group('Output Formats')
    formatting.add_argument('-s', '--summary', action='store_true',
                            help='Only generate report from overall '
                                 'test result.')

    formatting.add_argument('-f', '--format', default=TableConfig.PRETTY,
                            help='Output formats: ({0})'.format(
                                ', '.join(TableConfig.TYPES)))

    # -------------------------------

    files = parser.add_argument_group('Output Formats')

    files.add_argument('-o', '--output', default=None,
                       help='Name of output file')

    files.add_argument('-a', '--append', action='store_true',
                       help='Append output to existing file.')

    # -------------------------------

    debugging = parser.add_argument_group('Debugging')
    debugging.add_argument('-d', '--debug', action='store_true',
                           help='Set DEBUG flag to True.')

    debugging.add_argument('-t', '--target', default=None,
                           help='List data for specific TARGET test.')

    debugging.add_argument('-l', '--logfile', default=DEFAULT_LOG_FILE,
                           help='Name of log file. DEFAULT: {0}'.format(
                               DEFAULT_LOG_FILE
                           ))

    # -------------------------------

    return parser.parse_args()


if __name__ == '__main__':

    # Get Args
    args = parse_cli_args()

    # Setup logging (based on potential logging args)
    logging.basicConfig(
        filename=args.logfile, filemode="w", level=logging.INFO)
    log = logging.getLogger(__name__)

    # Setup Debug
    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug("DEBUG is enabled.")

    # Check for interdependent args; if reqs not met, error out.
    requires_outfile = [TableConfig.CSV, TableConfig.JSON]
    if args.format in requires_outfile and args.output is None:
        msg_fmt = ("\nERROR:\n\tNeed to specify output file (-o|--output) "
                   "for {0} file formats.\n")
        msg = (msg_fmt.format(
            ', '.join(["'{0}'".format(x) for x in requires_outfile])))

        log.error(msg)
        print(msg)

        exit(-1)

    if args.target is not None:
        log.info("Targeting Specific Test: {test}".format(test=args.target))

    # Parse results
    RallyResults = ParseRallyResults(
        rally_html_file=args.input, id_=args.id, summary_only=args.summary,
        target=args.target, debug=args.debug)

    # Generate Output
    results_table = RallYReportOutput(RallyResults)
    data_table = results_table.build_table(
        table_type=args.format, filename=args.output, append=args.append,
        summary_only=args.summary)

    log.info("\n{0}".format(data_table))
    print(data_table)
