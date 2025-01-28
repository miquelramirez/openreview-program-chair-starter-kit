from dataclasses import dataclass
from typing import *
from openreview.api import Note
from icaps.pcs import get_type_id, get_topic_id, get_mandatory_subject_tag_id, get_additional_subject_tag_id


@dataclass
class SubmissionStatus(object):
    id: str
    ratings: list[int]
    confidences: list[int]


def extract_submission(n: Note) -> dict:
    """
    Extracts submission data from OpenReview `Note` instance
    :param n:
    :return:
    """
    subject_tags: list = []

    for v in n.content['mandatory_subject_tags']['value']:
        subject_tags += [v]

    try:
        additional_tags = n.content['additional_subject_tags']['value']
        for v in additional_tags:
            subject_tags += [v]
    except KeyError:
        pass

    kernel: dict = dict(id=n.id,
                        number=n.number,
                        cdate=n.cdate,
                        title=n.content['title']['value'],
                        length=n.content['paper_length']['value'],
                        industry_paper=n.content['industry_paper']['value'],
                        student_paper=n.content['student_paper']['value'],
                        type=[v for v in n.content['type_of_contribution']['value']],
                        topic=[v for v in n.content['topic']['value']],
                        subject=subject_tags)
    return kernel