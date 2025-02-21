import json
import os
import shutil
import warnings
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Any, Optional

try:
    __version__ = version('insilico-aws')
except PackageNotFoundError:
    __version__ = 'dev'

RESOURCE_SCHEMA = '1.0.0'


def load_resources_definition() -> dict[str, Any]:
    resources_file = os.getenv(
        'INSILICO_AWS_RESOURCES',
        Path(__file__).parent / 'resources.json'
    )
    with open(resources_file, 'r') as f:
        resources = json.load(f)
    if RESOURCE_SCHEMA != resources.get('schema', ''):
        raise ValueError('Unsupported resource schema')
    return resources


def validate_parameters(schema: list[dict[str, Any]], inputs: Optional[dict[str, Any]]):
    if not inputs:
        return {}
    allowed_names = {k['Name'] for k in schema}
    user_names = {k for k in inputs}
    if unknown_params := user_names - allowed_names:
        warnings.warn(
            f"Params not supported: "
            f"{', '.join(unknown_params)}; "
            f"allowed: {', '.join(allowed_names)}"
        )
    for k, v in inputs.items():
        for p in schema:
            if p['Name'] != k:
                continue
            if not isinstance(v, {  # type: ignore
                'Integer': int,
                'Continuous': (float, int),
                'Categorical': str
            }[p['Type']]):
                raise ValueError(
                    f"Unsupported {k} param type, "
                    f"expected {p['Type']}, got {type(v).__name__}"
                )
            break
    return inputs


def load_examples(overwrite: bool = False):
    examples_dir = 'examples'
    return shutil.copytree(
        src=Path(__file__).parent / examples_dir,
        dst=Path.cwd() / examples_dir,
        dirs_exist_ok=overwrite
    )
