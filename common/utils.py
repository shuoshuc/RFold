import json
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


def spec_parser(specfile: str) -> dict:
    """
    Parse the cluster spec file.
    """
    with open(specfile, "r") as f:
        return json.load(f)


def dump_spec(spec: dict, specfile: str):
    """
    Dump the cluster spec to a file if `specfile` is specified.
    """
    if specfile:
        with open(specfile, "w") as f:
            json.dump(spec, f, indent=4)
