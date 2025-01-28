import json
from pathlib import Path
import time
import pandas as pd

import icaps
from icaps.pcs import get_type_id, get_topic_id, get_mandatory_subject_tag_id, get_additional_subject_tag_id

import openreview.api as api
from openreview import *


def extract_submission(n: Note) -> dict:
    """
    Extracts submission data from OpenReview `Note` instance
    :param n:
    :return:
    """
    subject_tags: list = []

    for v in n.content['mandatory_subject_tags']['value']:
        subject_tags += [get_mandatory_subject_tag_id(v)]

    try:
        additional_tags = n.content['additional_subject_tags']['value']
        for v in additional_tags:
            subject_tags += [get_additional_subject_tag_id(v)]
    except KeyError:
        pass

    author_ids: list = n.content['authorids']['value']
    has_unregistered_authors: bool = False
    for author in author_ids:
        if not author.startswith('~'):
            sub_id: str = n.id
            sub_number: int = n.number
            sub_title: str = n.content['title']['value']
            print(f"Submission: {sub_id} #{sub_number} with title: {sub_title} has unregistered authors")
            has_unregistered_authors = True
            break

    kernel: dict = dict(id=n.id,
                        number=n.number,
                        cdate=n.cdate,
                        status='Under_Review',
                        title=n.content['title']['value'],
                        length=n.content['paper_length']['value'],
                        has_unreg=has_unregistered_authors,
                        industry_paper=n.content['industry_paper']['value'],
                        student_paper=n.content['student_paper']['value'],
                        type=[get_type_id(v) for v in n.content['type_of_contribution']['value']],
                        topic=[get_topic_id(v) for v in n.content['topic']['value']],
                        subject=subject_tags)
    return kernel


if __name__ == "__main__":
    
    credentials: dict = icaps.load_credentials(Path() / "..")
    
    client = api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=credentials['username'],
            password=credentials['password']
        )

    # The venue id structure depends on how the conference request form is filled in
    venue_id: str = 'icaps-conference.org/ICAPS/2025/Conference'
    venue_group: api.Group = client.get_group(venue_id)

    submission_name: str = venue_group.content['submission_name']['value']
    submissions = client.get_all_notes(invitation=f'{venue_id}/-/{submission_name}')

    # Example of how to access to a submission
    # submission: Note = submissions[0]

    # We scan for people appearing in the author list that have not completed their registration tasks. These people
    # are of interest as if the registratio hasn't been completed then OpenReview will not be resolving conflicts of
    # interest for you.
    for sub in submissions:
        author_ids: list = sub.content['authorids']['value']
        has_unregistered_authors: bool = False
        for author in author_ids:
            if not author.startswith('~'):
                sub_id: str = sub.id
                sub_number: int = sub.number
                sub_title: str = sub.content['title']['value']
                print(f"Submission: {sub_id} #{sub_number} with title: {sub_title} has unregistered authors")
                has_unregistered_authors = True
                break
        if has_unregistered_authors:
            print(author_ids)
    
    # Columns of our "master" table of submissions, collecting all the important pieces of information that our customized
    # forms require.
    columns: list = ['id', 'status', 'number',  'length', 'has_unreg', 'industry_paper', 'student_paper', 'title', 'type', 'topic', 'subject']
    rows: list = []
    for entry in submissions:
        entry_data: dict = extract_submission(entry)
        rows += [[entry_data[k] for k in columns]]
    
    table: pd.DataFrame = pd.DataFrame(rows, columns=columns)

    # Query OpenReview for papers that have been withdrawn
    withdrawn_id = venue_group.content['withdrawn_venue_id']['value']
    withdrawn = client.get_all_notes(content={'venueid': withdrawn_id})

    # We update the table records corresponding to papers that have been withdrawn
    for submission in withdrawn:
        loc: pd.Index = table[table['number'] == submission.number].index
        table.loc[loc, 'status'] = 'Withdrawn'

    # Query OpenReview for papers that have been desk rejected
    desk_rejected_venue_id = venue_group.content['desk_rejected_venue_id']['value']
    desk_rejected = client.get_all_notes(content={'venueid': desk_rejected_venue_id})

    # As above, update their records accordingly
    for submission in desk_rejected:
        loc: pd.Index = table[table['number'] == submission.number].index
        table.loc[loc, 'status'] = 'Desk_Rejected'
    
    # Store the database on disk
    with open(Path() / ".." / "data" / "2025" / "papers_anonymised.json", "w") as output:
        json.dump(table.to_dict(), output, indent=4)