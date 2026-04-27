from pydantic import BaseModel
from pydantic_settings import CliApp, CliSubCommand

from dara.server.setting import DaraServerSettings


class Server(DaraServerSettings):
    """Run the Dara server in a Web UI."""

    def cli_cmd(self) -> None:
        from dara.server.app import launch_app
        launch_app()


class DaraCli(BaseModel):
    server: CliSubCommand[Server]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(DaraCli)
