__version__ = '1.0.0'


def setup():
    """
    Configure the settings (this happens as a side effect of accessing the
    first setting), configure logging
    """
    from wtb.commons.logging_conf import configure_logger

    configure_logger()


setup()
