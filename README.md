# A Starter Kit for PCs organising conferences on `openreview.net`

A bunch of scripts and examples using the OpenReview API that may prove useful to conference program chairs using OpenReview.

## Custom Forms

We customized almost all the `openreview.net` forms for ICAPS 2025. They can be found in the `forms` folder:

 - `decision_form.json` - The form used by OpenReview Decision stage.
 - `meta_review_form.json` - The form used by Area Chairs to enter their meta-reviews.
 - `review_form.json` - The form used by Reviewers to enter their reviews.
 - `submission_form.json` - Probably the most important form, used by Authors to describe their papers.
 - `reviewer_rego_form.json` - This form is used by the registration task for Reviewers. We used to collect important 
 information that was not captured directly by the OpenReview profiles.
 - `meta_reviewer_rego_form.json` - This form was used by the registration task for Area Chairs. As was the case of 
 Reviewers we asked Area Chairs for specific information about interests etc.

## Homepage Customization

It is possible to customize the look of the venue home page by entering a JSON document that describes its contents. 
We found quite difficult to figure out what was possible and significant trial and error was needed to get a decent
result. This can be found in the file `homepage/venue_homepage.json`.

## Scripts

OpenReview front-end is quite useful, but if you customize forms as we did, then you will need to make your own tools
to complement what the backend (exposed via OpenReview's API) can do for you. Below there is a collection of scripts we
found useful. We look forward to add more forms over time.

### Send message to all authors of active submissions

Script: `scripts/message_authors.py`

### Tabulate submissions with data collected with custom forms

Script: `scripts/retrieve_submissions.py`

### Replace an area chair or reviewer

Script: `scripts/replace_pc_member.py`
