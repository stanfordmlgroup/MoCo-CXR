import re


PATH_TO_STUDY_RE = re.compile(r'(valid|train|test)/patient(\d+)/study(\d+)')


def get_study_id(path):
    """Get a unique study ID from a (study or image) path.

    For example:
        /deep/group/xray4all/images/valid/patient64542/study1 -> valid/patient64542/study1

    Args:
        path (str): Path to convert to study_id.
    """
    path = str(path)
    match = PATH_TO_STUDY_RE.search(path)
    return match.group(0) if match else None
