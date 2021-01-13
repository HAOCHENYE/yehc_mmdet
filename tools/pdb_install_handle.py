import sys
def install_pdb_handler():
    """Signals to automatically start pdb:
      1. CTRL+\\ breaks into pdb.
      2. pdb gets launched on exception.
    """

    import signal
    import pdb

    def handler(_signum, _frame):
        pdb.set_trace()

    signal.signal(signal.SIGQUIT, handler)
    # signal.signal(signal.SIGINT, handler)

    # Drop into PDB on exception
    # from https://stackoverflow.com/questions/13174412
    def info(type_, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type_, value, tb)
        else:
            import traceback
            import pdb
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type_, value, tb)
            print()
            # ...then start the debugger in post-mortem mode.
            pdb.pm()

    sys.excepthook = info