###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

import argparse
import os
import sys

from SSMuLA import __version__


def print_help():
    lines = ['Helpful information']
    print('\n'.join(lines))


def main(args=None):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('something', type=str, help='a path to something')

    if len(sys.argv) == 1:
        print_help()
        sys.exit(0)
    elif sys.argv[1] in {'-v', '--v', '-version', '--version'}:
        print(f'SSMuLA v{__version__}')
        sys.exit(0)
    else:
        print(f'SSMuLA v{__version__}')
        args = parser.parse_args(args)

        # Validate the input arguments.
        if not os.path.isfile(args.something):
            print(f'The input file cannot be found: {args.something}')
            sys.exit(1)
        out_dir = os.path.dirname(args.output)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

    # Done - no errors.
    sys.exit(0)


if __name__ == "__main__":
    main()