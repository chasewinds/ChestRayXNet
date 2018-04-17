

import logging
import os

#
#  The following color-log come from <https://gist.github.com/KurtJacobson/c87425ad8db411c73c6359933e5db9f9>
#

from copy import copy
from logging import Formatter



class ColoredFormatter(Formatter):
    #
    # for shell color, refer to <https://misc.flogisoft.com/bash/tip_colors_and_formatting>
    #
    MAPPING = {
        'DEBUG': 35,  # Magenta
        'INFO': 32,  # green
        'WARNING': 33,  # yellow
        'ERROR': 31,  # red
        'CRITICAL': 27,  # reset blink
    }

    PREFIX = '\033['
    SUFFIX = '\033[0m'

    def __init__(self, patern, datefmt):
        self.datefmt = datefmt
        Formatter.__init__(self, patern, datefmt=datefmt)

    def format(self, record,):

        colored_record = copy(record)
        levelname = colored_record.levelname
        seq = self.MAPPING.get(levelname, 37) # default white
        colored_levelname = ('{0}{1}m{2}{3}') \
            .format(self.PREFIX, seq, levelname, self.SUFFIX)
        colored_record.levelname = colored_levelname

        # colored_record.msg = ('{0}{1}m{2}{3}') \
        #     .format(self.PREFIX, seq, record.msg, self.SUFFIX)
        return Formatter.format(self, colored_record)


#
# initialize log
#
def initlog(logpath="~/log", level=logging.INFO):
    import os, datetime
    logging.basicConfig(
        format='%(asctime)s(%(relativeCreated)d) - %(levelname)s %(filename)s(%(lineno)d) :: %(message)s',
        filename=os.path.expanduser(os.path.join(logpath, str(datetime.date.today())+'.txt')),
        filemode='a',
        level=logging.DEBUG,
    )
    formatter = logging.Formatter("[%(asctime)s(%(relativeCreated)d)] %(levelname)s\t%(message)s", datefmt='%m/%d/%y %H:%M:%S')
    colored_formatter = ColoredFormatter("[%(asctime)s(%(relativeCreated)d)] %(levelname)s\t%(message)s", datefmt='%m/%d/%y %H:%M:%S')

    console = logging.StreamHandler()
    console.setFormatter(colored_formatter)
    if os.getenv('DEBUG_TO_CONSOLE', '0') == '1':
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)