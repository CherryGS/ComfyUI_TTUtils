try:
    from rich import print, traceback

    traceback.install(show_locals=True)
except:
    pass
