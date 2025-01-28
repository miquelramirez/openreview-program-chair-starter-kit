import csv
import json
from pathlib import Path
import time
import pandas as pd
import icaps
import openreview.api as api
from openreview import *
from icaps.submissions import SubmissionStatus
from tqdm import tqdm

def main():
    credentials: dict = icaps.load_credentials()

    client = api.OpenReviewClient(
        baseurl='https://api2.openreview.net',
        username=credentials['username'],
        password=credentials['password']
    )

    venue_id: str = 'icaps-conference.org/ICAPS/2025/Conference'
    venue_group: api.Group = client.get_group(venue_id)

    submission_name = venue_group.content['submission_name']['value']
    submissions = client.get_all_notes(invitation=f'{venue_id}/-/{submission_name}', details='replies')

    # De-anonymise reviewer and AC profiles
    groups = client.get_all_groups(prefix=f'{venue_id}/Submission')
    rev_profiles = {group.id.split('/')[-1]: group.members[0] for group in groups if '/Reviewer_' in group.id}
    ac_profiles = {group.id.split('/')[-1]: group.members[0] for group in groups if '/Area_Chair_' in group.id}


    # Collect all reviews
    print("Retrieving reviews from OpenReview...")
    review_name = venue_group.content['review_name']['value']
    invitation = client.get_invitation(f'{venue_id}/-/{review_name}')
    content = invitation.edit['invitation']['edit']['note']['content']
    reviews = []
    print("\t processing submissions...")
    for s in tqdm(submissions):
        for reply in s.details['replies']:
            if f'{venue_id}/{submission_name}{s.number}/-/{review_name}' in reply['invitations']:
                reviews += [openreview.api.Note.from_json(reply)]

    keylist = list(content.keys())
    columns: list = ['paper.id', 'rev.id', ] + keylist
    rows: list = []

    print("\t processing reviews")
    for review in tqdm(reviews):
        reviewer_id: str = review.signatures[0].split('/')[-1]
        profile_id = rev_profiles[reviewer_id]

        rows += [[review.forum,
                  profile_id] + [review.content[k]['value'] for k in keylist]]

    reviews: pd.DataFrame = pd.DataFrame(rows, columns=columns)

    scores_table: pd.DataFrame = reviews[['paper.id', 'rating', 'confidence']]
    status: dict = {}
    for _, scoring in scores_table.iterrows():
        try:
            status[scoring['paper.id']].ratings += [scoring['rating']]
            status[scoring['paper.id']].confidences += [scoring['confidence']]
        except KeyError:
            status[scoring['paper.id']] = SubmissionStatus(id=scoring['paper.id'],
                                                           ratings=[scoring['rating']],
                                                           confidences=[scoring['confidence']])

    # Collect all papers under review
    print("Loading submissions cached data...")
    all_papers: pd.DataFrame = pd.read_json(Path() / "data" / "2025" / "papers_anonymous.json")
    reviewed_in_phase_1: pd.DataFrame = all_papers[all_papers['status'] == 'Under_Review']

    # Collect program committee
    print("Loading cached Phase 1 assignments...")
    program_committee_phase1: pd.DataFrame = pd.read_json(Path() / "data" / "2025" /"phase_1_updated_assignments.json")
    # We only need the ACs
    ac_assignments: pd.DataFrame = program_committee_phase1[program_committee_phase1['role'] == 'Area_Chairs']

    # Load rejection candidates reviews
    phase_1_review_outcome: pd.DataFrame = pd.read_csv(Path() / "data" / "2025" / "phase_1_decisions_review.csv")

    # Generate decisions table for OpenReview
    print("Generating decisions table")
    columns: list = ['paper_number', 'decision', 'comment']
    rows: list = []
    for _, data in tqdm(reviewed_in_phase_1.iterrows()):
        review_outcome = phase_1_review_outcome[phase_1_review_outcome['id'] == data['id']]
        if len(review_outcome) == 0:
            continue # Paper goes onto Phase 2
        if review_outcome['Outcome'].values[0] != 'Reject':
            continue
        comment: str = "Paper rejected by recommendation of the Reviewers and AC."
        decision_data = [data['number'], 'Reject', f'"{comment}"']

        rows += [decision_data]

    rejections: pd.DataFrame = phase_1_review_outcome[phase_1_review_outcome['Outcome'] == 'Reject']

    print("Writing CSV file with decisions data for OpenReview...")
    open_review_decisions_path: Path = Path() / "data" / "2025" / "phase_1_decisions.csv"

    phase_1_decisions: pd.DataFrame = pd.DataFrame(rows, columns=columns)
    phase_1_decisions.to_csv(open_review_decisions_path, index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)

    with open(Path() / "data" / "2025" / "phase_1_decisions.json", "w") as f:
        json.dump(phase_1_decisions.to_dict(), f, indent=4)

if __name__ == "__main__":
    main()