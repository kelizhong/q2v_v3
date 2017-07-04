import argparse
import sys


def set_default_subparser(self, name, args=None):
    """default subparser selection. Call after setup, just before parse_args()
    name: is the name of the subparser to call by default
    args: if set is the argument list handed to parse_args()

    , tested with 2.7, 3.2, 3.3, 3.4
    it works with 2.6 assuming argparse is installed
    """
    subparser_found = False
    existing_default = False # check if default parser previously defined
    for arg in sys.argv[1:]:
        if arg in ['-h', '--help']:  # global help if no subparser
            break
    else:
        for x in self._subparsers._actions:
            if not isinstance(x, argparse._SubParsersAction):
                continue
            for sp_name in x._name_parser_map.keys():
                if sp_name in sys.argv[1:]:
                    subparser_found = True
                if sp_name == name: # check existance of default parser
                    existing_default = True
        if not subparser_found:
            # If the default subparser is not among the existing ones,
            # create a new parser.
            # As this is called just before 'parse_args', the default
            # parser created here will not pollute the help output.

            if not existing_default:
                for x in self._subparsers._actions:
                    if not isinstance(x, argparse._SubParsersAction):
                        continue
                    x.add_parser(name)
                    break # this works OK, but should I check further?

            # insert default in first position, this implies no
            # global options without a sub_parsers specified
            if args is None:
                sys.argv.insert(1, name)
            else:
                args.insert(0, name)

argparse.ArgumentParser.set_default_subparser = set_default_subparser
