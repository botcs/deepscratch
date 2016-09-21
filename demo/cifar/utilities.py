import sys
import os
import warnings


def ensure_dir(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError as e:
            print ('Cannot make directory: ' + str(e))
            raise


def warning(message, instance, warn_type=FutureWarning):
    note = '\n  Warning for: ' + repr(instance)
    warnings.warn(message + note, warn_type)


class StatusBar:

    def __init__(self, total, barLength=30):
        self.total = total
        self.curr = 0
        self.percentage = 0
        self.barLength = barLength

    def barStr(self):
        currBar = self.barLength * self.percentage / 100
        return '[' + "=" * currBar + " " * (self.barLength - currBar) + ']'

    def printBar(self, msg):
        if(self.percentage <= 100):
            print("\r  " + self.barStr() + " (" +
                  str(self.curr) + '/' + str(self.total) + ")   " +
                  str(100 * self.curr / self.total) +
                  "%  {}   ".format(msg)),
            sys.stdout.flush()
            if(self.percentage == 100):
                print '\n'

    def update(self, msg):
        self.curr += 1
        currPercentage = self.curr * 100 / self.total
        if(currPercentage > self.percentage):
            self.percentage = currPercentage
            self.printBar(msg)
