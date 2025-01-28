import os
import json
import time
import pandas as pd
import numpy as np

from argparse import ArgumentParser, Namespace
from pathlib import Path
from pathlib import Path

import icaps
import icaps.pcs as pcs
import openreview.api as api

from openreview import *
from tqdm import tqdm


def process_cmd_line_args() -> Namespace:
    """
    Process command line arguments and return a Namespace object.
    """
    parser: ArgumentParser = ArgumentParser(description="Adds Reviewers to their assigned papers forums...")
    parser.add_argument("--input-file",
                        type=str,
                        default=None,
                        required=True,
                        help="Input CSV file containing the assignment data")
    opt: Namespace = parser.parse_args()
    opt.input_file = Path() / opt.input_file
    if not opt.input_file.exists():
        raise SystemExit(f"Input file {opt.input_file} does not exist")

    return opt


def main(opt: Namespace) -> None:
    """
    Main entry point of the script
    """
    assignments: pd.DataFrame = pd.DataFrame.read_csv(opt.input_file)
    credentials: dict = icaps.load_credentials(Path() / "..")
    client = api.OpenReviewClient(
        baseurl='https://api2.openreview.net',
        username=credentials['username'],
        password=credentials['password'])
    venue_id: str = 'icaps-conference.org/ICAPS/2025/Conference'
    venue_group: api.Group = client.get_group(venue_id)

    role_id: str = 'Reviewers'
    print(f"Adding reviewers to each paper forum...")
    for _, data in tqdm(assignments.iterrows()):
        number: str = data['paper.number']
        rev: str = data['rev.id']
        print(f'{venue_id}/Submission{number}/{role_id} <- {rev}')
        client.add_members_to_group(group=f'{venue_id}/Submission{number}/{role_id}', members=[data['rev.id']])


if __name__ == "__main__":
    main(process_cmd_line_args())