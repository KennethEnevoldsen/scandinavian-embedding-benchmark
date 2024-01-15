# This just created a rough draft of the CLI documentation. It is not
# intended to be used for anything other than a starting point.
# at least we would need this issue fixed first:
# https://github.com/explosion/radicli/issues/30

from pathlib import Path

from seb.cli import cli

title = "Command Line Interface"
description = "Documentation for the command line interface of SEB."

if __name__ == "__main__":
    with Path("docs/cli.md").open("w", encoding="utf8") as f:
        f.write(cli.document(title=title, description=description))
