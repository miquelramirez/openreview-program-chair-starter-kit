
import json
from pathlib import Path
import time
import pandas as pd
import icaps.pcs as pcs
import numpy as np
import icaps
import openreview.api as api
from openreview import *
from dataclasses import dataclass

@dataclass
class AssignmentEdit:
    """
    Simple dataclass to encapsulate the data required to describe a paper reassignment
    """
    paper_id: str
    paper_number: int
    orig_profile: str
    new_profile: str
    # Note that this field needs to be filled when the role in question is other than Reviewer, e.g. 'Area_Chairs' or
    # 'Senior_Area_Chairs'
    role_id: str = 'Reviewers'

if __name__ == '__main__':

    # For the purpose of illustration, I have hardcoded the edits in a list. This could be loaded from a CSV, JSON, whatever
    assignment_edits = [
        AssignmentEdit(paper_id="HQK3daUQZ", paper_number=1077, orig_profile="~Troy_Freeman1", new_profile="~Ava_Morton1"),
        AssignmentEdit(paper_id="5hlH39z9yV", paper_number=944, orig_profile="~William_Chasey1", new_profile="~Lois_Maldonado1"),
        AssignmentEdit(paper_id="sEWJ5mPhGE", paper_number=253, orig_profile="~Gresham_Norman1", new_profile="~Kenneth_Knight1"),
        AssignmentEdit(paper_id="3WgFx0o1SD", paper_number=111, orig_profile="~Ursa_Adkins1", new_profile="~Courtney_Maldonado1"),
        AssignmentEdit(paper_id="Ya9bs3bPNF", paper_number=37, orig_profile="~Hattie_Pope1", new_profile="~Grace_Whitehead1"),
    ]

    # Logging in yada, yada
    credentials: dict = icaps.load_credentials(Path() / "..")
    client = api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=credentials['username'],
            password=credentials['password']
        )

    # We locate the objects of interest (assignments)
    venue_id: str = 'icaps-conference.org/ICAPS/2025/Conference'
    venue_group: api.Group = client.get_group(venue_id)

    for entry in assignment_edits:
        print(f"Reassigning paper #{entry.paper_number}...")

        # We first remove from the group associated with the role the person to replaced
        client.remove_members_from_group(group=f'{venue_id}/Submission{entry.paper_number}/{entry.role_id}',
                                         members=[entry.orig_profile])

        # We delete the tuple from the "Assignment" relation that associates the person to be replaced with the paper
        client.delete_edges(invitation=f'{venue_id}/{entry.role_id}/-/Assignment',
                            head=entry.paper_id,
                            tail=entry.orig_profile,
                            wait_to_finish=True)

        # We create the new tuple reflecting the relation between the replacement person and the paper
        edge = openreview.api.Edge(invitation=f'{venue_id}/{entry.role_id}/-/Assignment',
                                   head=entry.paper_id,
                                   tail=entry.new_profile,
                                   signatures=[f'{venue_id}/Program_Chairs'], weight=1)

        # We post the change to the database
        openreview.tools.post_bulk_edges(client=client, edges=[edge])

        # Then we update the corresponding group with the profile id of the replacement
        client.add_members_to_group(group=f'{venue_id}/Submission{entry.paper_number}/{entry.role_id}',
                                    members=[entry.new_profile])

