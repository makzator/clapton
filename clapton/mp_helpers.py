import psutil
import os
import signal


class SignalHandler:
    def __init__(self):
        self.signals = [signal.SIGINT, signal.SIGTERM]
        self.default_handlers = [signal.getsignal(s) for s in self.signals]
        self.set_custom_handler()
    def custom_handler(signum, frame):
        pid = os.getpid()
        killtree(pid, False)
        raise Exception(f"Caught signal {signum}. Killed all subprocesses. Terminated.")
    def set_handler(self, handler):
        for s in self.signals:
            signal.signal(s, handler)
    def set_custom_handler(self):
        self.set_handler(SignalHandler.custom_handler)
    def restore_handlers(self):
        for i in range(len(self.signals)):
            signal.signal(self.signals[i], self.default_handlers[i])

def killtree(pid, including_parent=True):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        print(f"killed child {child}")
        child.kill()
    if including_parent:
        print(f"killed parent {parent}")
        parent.kill()