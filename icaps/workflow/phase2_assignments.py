import ortools as ot
from ortools.sat.python import cp_model
import numpy as np
import pandas as pd
import json
from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm
from typing import *

import time


class SolutionCallbackHandler(cp_model.CpSolverSolutionCallback):
    """Collects intermediate solutions."""

    def __init__(self, vars: list[cp_model.BoolVarT]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.soln_count = 0
        self.vars = vars
        self.t0 = time.time()

    @property
    def solution_count(self) -> int:
        return self.soln_count

    def on_solution_callback(self):
        self.soln_count += 1
        t_now: float = time.time()
        print(f"Found suboptimal solution #{self.soln_count}, elapsed {t_now - self.t0}s")
        if t_now - self.t0 >= 300:
            self.stop_search()
        # if self.soln_count >= 1 :
        #    self.stop_search()


def calc_phase_2_papers() -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Calculates the set of papers that are eligible for reviews in Phase 2
    """
    print("Calculating phase 2 papers...")

    all_papers: pd.DataFrame = pd.read_json(Path() / "data" / "2025" / "papers_anonymous.json")
    reviewed_in_phase_1: pd.DataFrame = all_papers[all_papers['status'] == 'Under_Review']
    phase_1_decisions: pd.DataFrame = pd.read_json(Path() / "data" / "2025" / "phase_1_decisions.json")
    for index, data in tqdm(reviewed_in_phase_1.iterrows()):
        paper_number: int = data['number']
        if paper_number in phase_1_decisions['paper_number'].values:
            reviewed_in_phase_1.at[index, 'status'] = 'Phase_1_Rejection'

    columns: list = ['id', 'title', 'number', 'type', 'topic', 'subject']
    rows: list = []
    for index, data in tqdm(reviewed_in_phase_1.iterrows()):
        if data['status'] == 'Under_Review':
            rows += [[data['id'], data['title'], data['number'], data['type'], data['topic'], data['subject']]]

    print("Indexing papers...")
    phase_2_under_review: pd.DataFrame = pd.DataFrame(rows, columns=columns)
    paper_indexer: dict = {}
    for index, data in tqdm(phase_2_under_review.iterrows()):
        paper_indexer[data['id']] = index

    return phase_2_under_review, paper_indexer, reviewed_in_phase_1


def main() -> None:
    """
    The main function
    """
    res: tuple  = calc_phase_2_papers()
    phase_2_under_review: pd.DataFrame = res[0]
    paper_indexer: dict = res[1]
    reviewed_in_phase_1: pd.DataFrame = res[2]

    print("Calculating team of reviewers...")
    phase_1_assignments: pd.DataFrame = pd.read_json(Path() / "data" / "2025" / "phase_1_updated_assignments.json")
    phase_1_assignments = phase_1_assignments[phase_1_assignments['role'] == 'Reviewers'][["profile", "paper.id"]]

    columns: list = ['profile', 'paper.id']
    rows: list = []
    for _, data in phase_1_assignments.iterrows():
        if data['paper.id'] in reviewed_in_phase_1['id'].values:
            rows += [[data['profile'], data['paper.id']]]
    phase_1_assignments: pd.DataFrame = pd.DataFrame(rows, columns=columns)

    phase_1_assignment_hash: set = set(
        [(data['profile'], data['paper.id']) for _, data in phase_1_assignments.iterrows()])

    phase_1_blacklist: pd.DataFrame = pd.read_csv(Path() / "data" / "phase_1_reviewer_blacklist.csv")

    rev_team: pd.DataFrame = pd.read_json(Path() / "data" / "2025" / "verified_rev_profiles.json")
    columns = ['id', 'active', 'type', 'topic', 'subject']
    rows = []
    for index, row in rev_team.iterrows():
        rows += [[row['id'], row['active'],
                  [int(t) for t in row['type']],
                  [int(t) for t in row['topic']],
                  [int(t) for t in row['subject']]]]
    rev_team = pd.DataFrame(rows, columns=columns)

    nominated_assignments: pd.DataFrame = pd.read_csv(Path() / "data" / "2025" / "phase_2_reviewer_nominations.csv")
    blocked_assignments: pd.DataFrame = pd.read_csv(Path() / "data" / "2025" / "phase_2_blocked_assignments.csv")
    extra_reviewers: pd.DataFrame = pd.read_csv(Path() / "data" / "2025" / "extra_reviews.csv")
    needs_4_reviewers: set = set(extra_reviewers['paper.id'].values)

    rev_indexer: dict = {}

    for index, data in rev_team.iterrows():
        rev_indexer[data['id']] = index

    rev_bids: pd.DataFrame = pd.read_json(Path() / "data" / "2025" / "rev_bids.json")
    rev_scores: pd.DataFrame = pd.read_json(Path() / "data" / "2025" / "rev_scores.json")
    rev_conflicts: pd.DataFrame = pd.read_json(Path() / "data" / "2025" / "rev_conflicts.json")

    # Construct cost matrices
    nP: int = len(phase_2_under_review)
    nR: int = len(rev_team)
    print("Constructing cost matrices...")

    print("\t Bid component...")
    # Note initialization to high cost
    bid_cost = 100 * np.ones((nP, nR), dtype=np.int64)
    # Overwrite bid costs with actual values
    for _, data in rev_bids.iterrows():
        try:
            paper: int = paper_indexer[data['paper.id']]
        except KeyError:
            continue
        try:
            person: int = rev_indexer[data['rev.id']]
        except KeyError:
            continue
        bid_cost[paper, person] = 10 * data['label']

    print("\t Affinity component...")
    # Note also how we initialize the scores here to very large numbers
    affinity_cost = 100 * np.ones((nP, nR), dtype=np.int64)

    # Overwrite affinity costs with actual values
    for _, data in rev_scores.iterrows():
        try:
            paper: int = paper_indexer[data['paper.id']]
        except KeyError:
            continue
        try:
            person: int = rev_indexer[data['rev.id']]
        except KeyError:
            continue  # Ignore inactive profiles
        affinity_cost[paper, person] = int(data['score'])

    print("\t Keyword component...")
    keyword_cost = np.zeros((nP, nR), dtype=int)
    for rev_id, rev_data in rev_team.iterrows():
        for paper_id, paper_data in phase_2_under_review.iterrows():
            keyword_cost[paper_id, rev_id] = 0
            for t in paper_data['type']:
                if t not in rev_data['type']:
                    keyword_cost[paper_id, rev_id] += 5
            for topic in paper_data['topic']:
                if topic not in rev_data['topic']:
                    keyword_cost[paper_id, rev_id] += 10
            for subj in paper_data['subject']:
                if subj not in rev_data['subject']:
                    keyword_cost[paper_id, rev_id] += 15
            # Clamp this value to 100 inline with the rest of the scores
            keyword_cost[paper_id, rev_id] = min(100, keyword_cost[paper_id, rev_id])

    print("\t Trust component...")
    trust_cost = np.zeros((nP, nR), dtype=int)
    for rev_id, rev_data in rev_team.iterrows():
        person: int = rev_indexer[rev_data['id']]
        cost: int = 100
        if rev_data['active']:
            cost = 0
        for paper_id, paper_data in phase_2_under_review.iterrows():
            paper: int = paper_indexer[paper_data['id']]
            trust_cost[paper, person] = cost

    print("\t Combining components...")
    costs = bid_cost + affinity_cost + keyword_cost + trust_cost
    relevance_cost = affinity_cost + keyword_cost


    ignored_assignments: list = [
        '~Fabio_Patrizi1', # AC that seconded as Reviewer
        '~Andre_Augusto_Cire1', # Changed his name
        '~Anton_Andreychuk1', # Came in kind of late
        '~Miquel_Ramirez1' # That's me
    ]
    rev_indexer['~Andre_Augusto_Cire1'] = rev_indexer['~Andre_Cire1']
    excused = [
        '~Laura_Sebastia1' # Impacted by Valencia's floods
    ]

    # Size of the PC during Phase 1
    pc_size = np.zeros(nP, dtype=np.int64)
    # Counting papers assigned to this person in Phase 1
    for _, data in phase_1_assignments.iterrows():
        #r: int = rev_indexer[data['profile']]
        try:
            p: int = paper_indexer[data['paper.id']]
        except KeyError:
            continue
        pc_size[p] += 1
    print(pc_size)

    print("Constructing CP model...")
    # CP Model setup
    model = cp_model.CpModel()
    vars = np.ndarray((nP, nR), dtype=object)

    print("\t Variables...")
    for rev_id, rev_data in rev_team.iterrows():
        for paper_id, paper_data in phase_2_under_review.iterrows():
            x_ij = model.new_bool_var(name=f'x_{paper_id}_{rev_id}')
            vars[paper_id, rev_id] = x_ij

    print("\t Constraints - reviewer capacity constraints...")
    for rev_id, rev_data in rev_team.iterrows():
        r: int = rev_indexer[rev_data['id']]
        if rev_data['id'] in phase_1_blacklist['profile'].values \
                or rev_data['id'] in excused:
            # Blacklisted reviewer reviews nothing
            model.add_linear_constraint(np.sum(vars[:, r]), 0, 0)
            continue
        # Maximum total assignment for a reviewer is 4 - assignments in phase 1
        #model.add_linear_constraint(np.sum(vars[:, r]), 0, 4 - lb[r])
        #model.add_linear_constraint(np.sum(vars[:, r]), 0, 5 - lb[r])
        model.add_linear_constraint(np.sum(vars[:, r]), 0, 2)

    print("\t Constraints - required reviews by each paper...")
    for paper_id, paper_data in phase_2_under_review.iterrows():
        p: int = paper_indexer[paper_data['id']]
        if paper_data['id'] in needs_4_reviewers:
            model.add_linear_constraint(np.sum(vars[paper_id, :]), 4 - pc_size[p], 4 - pc_size[p])
            continue
        model.add_linear_constraint(np.sum(vars[paper_id, :]), 3 - pc_size[p], 3 - pc_size[p])

    print("\t Constraints - calculating conflicts of interest...")
    # Conflicts of interest
    cois: list = []
    cois_hash: set = set()
    for _, data in rev_conflicts.iterrows():
        try:
            r: int = rev_indexer[data['rev.id']]
        except KeyError:
            continue
        try:
            paper: int = paper_indexer[data['paper.id']]
        except KeyError:
            continue
        cois += [vars[paper, r].Not()]
        cois_hash.add((paper, r))

    # Manual conflicts of interest, due to folks not ahaving OpenReview profiles etc.
    print("\t Constraints - Blocked assignments...")

    for _, data in blocked_assignments.iterrows():
        r: int = rev_indexer[data['reviewer.id']]
        try:
            paper: int = paper_indexer[data['paper.id']]
        except KeyError:
            continue
        cois += [vars[paper, r].Not()]

    print("\t Constraints - Nominated reviewers...")
    fixed_assignments: list = []
    count: int = 0
    for _, data in nominated_assignments.iterrows():
        r: int = rev_indexer[data['reviewer.id']]
        p: int = paper_indexer[data['paper.id']]
        if (p, r) in cois_hash:
            raise RuntimeError("Nominated assignment invalid as it overlaps with COIs: paper:", data['paper.id'], "reviewer:", data['reviewer.id'])
        fixed_assignments += [vars[p, r]]
        count += 1

    # these are all people who have been discarded or are standing in
    # for someone else
    ignored_assignments: list = [
        '~Fabio_Patrizi1',
        '~Anton_Andreychuk1',
        '~Miquel_Ramirez1'
    ]
    print("\t Constraints - Phase 1 assignments...")
    for _, data in phase_1_assignments.iterrows():
        if data['profile'] in ignored_assignments:
            continue
        if data['profile'] in phase_1_blacklist['profile'].values or data['profile'] in excused:
            continue  # do not fix the assignment it is already zeroed
        r: int = rev_indexer[data['profile']]
        try:
            p: int = paper_indexer[data['paper.id']]
        except KeyError:
            continue
        fixed_assignments += [vars[p, r].Not()]

    model.add_assumptions(cois + fixed_assignments)
    print("\t Objective function...")
    model.minimize(np.sum(np.multiply(costs, vars).flatten()))

    print("Calling CP solver...")
    cb = SolutionCallbackHandler(vars)
    solver = cp_model.CpSolver()
    status = solver.solve(model, cb)

    columns = ['rev.id', 'paper.id', 'paper.number', 'paper.title', 'phase', 'bid', 'affinity', 'keywords', 'trust']
    rows = []
    print(f"\tSolver result: {status}")
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Total cost = {solver.objective_value}\n")
        for rev_id, rev_data in rev_team.iterrows():
            for paper_id, paper_data in phase_2_under_review.iterrows():
                if solver.boolean_value(vars[paper_id, rev_id]):
                    phase = 2
                    row = [rev_data['id'], paper_data['id'], paper_data['number'], paper_data['title'],
                           phase,
                           bid_cost[paper_id, rev_id],
                           affinity_cost[paper_id, rev_id],
                           keyword_cost[paper_id, rev_id],
                           trust_cost[paper_id, rev_id]]
                    rows += [row]
                    # print(f"{rev_data['id']} -> {paper_data['number']}, title: {paper_data['title']}")

        assignment0: pd.DataFrame = pd.DataFrame(rows, columns=columns)

        assignment0.to_csv('assignment0.csv', index=False, header=True)
    elif status == cp_model.INFEASIBLE:
        print("No solution found")
    else:
        print("Something is wrong, check the status and the log of the solve")



if __name__ == '__main__':

    main()

