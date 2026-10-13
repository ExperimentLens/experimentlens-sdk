import pytest

from experimentlens.judge import _parse_verdict
from experimentlens.llm import messages_to_prompt


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"passed": true, "rationale": "Correct."}', (True, "Correct.")),
        ('Sure! ```json\n{"passed": false, "rationale": "Wrong city."}\n```', (False, "Wrong city.")),
        ("PASSED - the answer names Paris.", (True, "PASSED - the answer names Paris.")),
        ("This fails the criteria.", (False, "This fails the criteria.")),
        ("Passed? No, it failed.", (False, "Passed? No, it failed.")),
    ],
)
def test_parse_verdict(text, expected):
    assert _parse_verdict(text) == expected


def test_messages_to_prompt():
    assert messages_to_prompt([{"role": "system", "content": "Be brief."}, {"content": "Hi"}]) == (
        "system: Be brief.\n\nuser: Hi"
    )
