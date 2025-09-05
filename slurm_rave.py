from pathlib import Path
import enum
from string import Template
import subprocess
from typing import List
import re

import click
import questionary

VERSION = "0.1.0"

RAVE_JOBS_DIR = Path("~/rave_jobs").expanduser()
RAVE_JOBS_DIR.mkdir(exist_ok=True, parents=True)
RAVE_BIN = Path("~/slurm_rave/.venv/bin/rave").expanduser()


class ModelVersion(enum.Enum):
    v2 = "v2"
    v3 = "v3"


RAVE_DATASET_SCRIPT = r"""
set -e

PATH="$PATH":/home/gx547144/ffmpeg-git-20240629-amd64-static {COMMAND}
"""


class RaveDataset:
    """The dataset gets rendered by our RAVE environment."""

    def __init__(self, paths: List[Path], name: str, out_dir: Path):
        self.paths = paths
        self.name = name
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def build_dataset(self):
        """Prepares the building a dataset command."""
        click.echo("Generating the dataset...")

        args: List[str] = []

        args.extend([str(RAVE_BIN.absolute()), "preprocess"])
        for path in self.paths:
            args.extend(["--input_path", str(path.absolute())])
        args.extend(["--output_path", str(self.out_dir.absolute())])

        script = RAVE_DATASET_SCRIPT.format(
            COMMAND=" ".join([str(arg) for arg in args])
        )

        print(script)
        try:
            subprocess.check_output(
                script,
                shell=True,
                executable="/usr/bin/zsh",
                text=True,
                env={},
            )
        except subprocess.CalledProcessError as e:
            click.echo(
                click.style(
                    f"Dataset generation failed: {e.returncode}: {e.stderr}", fg="red"
                )
            )
            raise e

        click.echo(
            click.style(
                f"Successfully generated the dataset ({self.out_dir})", fg="green"
            )
        )


RAVE_SCRIPT_TEMPLATE = r"""#!/usr/bin/zsh

############################################################
### Slurm flags
############################################################

#SBATCH --partition=c23g            # request partition with GPU nodes
#SBATCH --nodes=1                   # request desired number of nodes
#SBATCH --ntasks-per-node=1         # request desired number of processes (or MPI tasks)

#SBATCH --cpus-per-task=8          # request desired number of CPU cores or threads per process (default: 1)

#SBATCH --gres=gpu:1                # specify desired number of GPUs per node
#SBATCH --time=72:00:00             # max. run time of the job
#SBATCH --job-name=@JOB_NAME    # set the job name
#SBATCH --output=stdout_@JOB_NAME%j.txt      # redirects stdout and stderr to stdout.txt
#SBATCH --account=rwth1852

############################################################
### Parameters and Settings
############################################################

# load modules
module load GCCcore/11.3.0
module load Python/3.10.4

# print some information about current system
echo "Job nodes: ${SLURM_JOB_NODELIST}"
echo "Current machine: $(hostname)"
nvidia-smi

############################################################
### Setup
############################################################

source ~/venv/bin/activate

rave train --config @MODEL_VERSION --db_path @DATASET_DIR --out_path @OUT_DIR --max_steps 4000000 --name @JOB_NAME --channels 1

rave export --run @OUT_DIR --fidelity 0.99 --name @JOB_NAME --output @EXPORT_DIR
"""


class RaveTemplate(Template):
    delimiter = "@"


class RaveModel:
    def __init__(
        self,
        job_name: str,
        model_version: ModelVersion,
        dataset_dir: Path,
        project_dir: Path,
    ):
        self.job_name = job_name
        self.model_version = model_version
        self.dataset_dir = dataset_dir
        self.project_dir = project_dir
        self.model_version = model_version

    def send_job_to_slurm(self):
        out_dir = self.project_dir.joinpath("model")
        out_dir.mkdir(exist_ok=True, parents=True)

        export_dir = self.project_dir.joinpath("export")
        export_dir.mkdir(exist_ok=True, parents=True)

        script = RaveTemplate(RAVE_SCRIPT_TEMPLATE).substitute(
            {
                "DATASET_DIR": self.dataset_dir,
                "OUT_DIR": out_dir,
                "EXPORT_DIR": export_dir,
                "MODEL_VERSION": self.model_version,
                "JOB_NAME": self.job_name,
            }
        )

        shell_script_path = self.project_dir.joinpath("job.sh")

        with shell_script_path.open("w") as f:
            f.write(script)

        subprocess.run(["chmod", "+x", shell_script_path.absolute()], check=True)

        try:
            out = subprocess.check_output(
                ["sbatch", str(shell_script_path.absolute())],
                cwd=self.project_dir,
            )
            click.echo(out)
        except subprocess.CalledProcessError as e:
            click.echo(
                click.style(
                    f"Failed to submit slurm job: {e.returncode}: {e.stderr}", fg="red"
                )
            )
            raise e
        click.echo(click.style("Successfully submitted job to SLURM", fg="green"))


def get_dataset_dirs(wav_paths: List[Path]) -> List[Path]:
    continue_asking = len(wav_paths) <= 0
    while continue_asking:
        wav_path = questionary.path(
            "Add a directory with wave files",
            only_directories=True,
        ).ask()

        wav_path_name = Path(wav_path)
        if wav_path_name.is_dir():
            click.echo(f"Adding {wav_path_name.absolute()} to dataset")
            wav_paths.append(wav_path_name.absolute())
            continue_asking = questionary.confirm("Add another directory?").ask()
        else:
            click.echo(click.style("This is not a valid directory", fg="red"))
    return wav_paths


def get_job_name(job_name: str, project_base_path: Path) -> str:
    job_name = (
        job_name or ""
    )  # just to get sure that we are actually working on a string...
    job_regex = re.compile(r"^[a-zA-Z0-9-_]+$")

    while True:
        job_name = questionary.text(
            "What is the name of the job?", instruction="Use only A-z 0-9 and -_"
        ).ask()
        if job_regex.match(job_name) is None:
            click.echo(click.style("This is not a valid job name!", fg="red"))
        elif project_base_path.joinpath(job_name).exists():
            click.echo(click.style("Project directory already exists", fg="red"))
        else:
            break

    return job_name


@click.command()
@click.option(
    "--wav-path",
    "-p",
    "wav_paths",
    multiple=True,
    help="Directory containing WAV files - allows multiple mentions",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--rave-version",
    help="Rave architecture version to use (v2/v3)",
    type=ModelVersion,
)
@click.option(
    "--name",
    "job_name",
    help="Name of the RAVE model and slurm job",
)
@click.option(
    "--project-dir",
    "project_base_path",
    help="Directory where the generated RAVE project files should be stored",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=RAVE_JOBS_DIR,
)
def main(
    wav_paths: List[Path],
    rave_version: ModelVersion,
    job_name: str,
    project_base_path: Path,
):
    wav_paths = list(wav_paths)  # hack: click actually returns a tuple here...
    click.echo(
        click.style(f"Welcome to slurm_rave {VERSION}", bold=True, fg="red", bg="green")
    )

    job_name = get_job_name(job_name, project_base_path)
    wav_paths = get_dataset_dirs(wav_paths)

    project_path = project_base_path.joinpath(job_name)
    click.echo(f"Use project dir {project_path}")

    dataset_path = project_path.joinpath("dataset")
    rave_dataset = RaveDataset(
        paths=wav_paths,
        name=job_name,
        out_dir=dataset_path,
    )
    rave_dataset.build_dataset()

    if not rave_version:
        rave_version = questionary.select(
            "Select a model architecture", choices=ModelVersion._member_names_
        ).ask()
    click.echo(click.style(f"Using model {rave_version}"))

    rave_model = RaveModel(
        job_name=job_name,
        model_version=rave_version,
        dataset_dir=dataset_path,
        project_dir=project_path,
    )
    rave_model.send_job_to_slurm()


if __name__ == "__main__":
    main()
