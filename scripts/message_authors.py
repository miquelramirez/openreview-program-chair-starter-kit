import json
from pathlib import Path
import time
import pandas as pd
import icaps
from tqdm import tqdm

import openreview.api as api
from openreview import *


if __name__ == "__main__":

    # Loads OpenReview credentials from a JSON document
    credentials: dict = icaps.load_credentials(Path() / "..")

    client = api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=credentials['username'],
            password=credentials['password']
        )

    venue_id: str = 'icaps-conference.org/ICAPS/2025/Conference'
    venue_group: api.Group = client.get_group(venue_id)

    submissions = client.get_all_notes(invitation = f'{venue_id}/-/Submission')
    for submission in tqdm(submissions):
        subject = f"[ICAPS 2025] Authors' rebuttal period delayed"
        message = f"""\
Dear {{{{fullname}}}},

Thank you for submitting your work to ICAPS 2025. We love you all very much.

Best regards,
Daniel and Miquel"""
        recipients = submission.content['authorids']['value']
        invitation = f'{venue_id}/-/Edit'
        client.post_message(subject, recipients, message, invitation=invitation, signature=f'{venue_id}/Program_Chairs')

