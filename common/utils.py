import logging
import simpy


class Signal:
    """
    A wrapper class for simpy events.
    """

    def __init__(self, env: simpy.core.Environment):
        self.env = env
        self.event = env.event()

    def trigger(self):
        """
        Trigger once and reloads the event.
        """
        self.event.succeed()
        self.event = self.env.event()

    def signal(self):
        """
        Expose the underlying event.
        """
        return self.event


class PrettyForm(logging.Formatter):
    """
    Custom log formatter to make padding and alignment easier.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record):
        record.module = f"{record.module}::{record.funcName}():{record.lineno}"
        return super().format(record)
