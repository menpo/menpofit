import nose
import sys


if nose.run(argv=['', 'menpofit']):
    sys.exit(0)
else:
    sys.exit(1)
