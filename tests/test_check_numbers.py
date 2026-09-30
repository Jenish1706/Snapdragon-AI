import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_numbers as cn


def test_numbers_normalises_signs_and_thousands():
    assert cn.numbers("from 21.88 to 50,360 (+230,019 %), −0.39 % and 4.41") == [
        "21.88", "50360", "+230019", "-0.39", "4.41"]


def test_split_items_keys_numbered_and_bulleted_items_by_their_bold_title():
    block = "Intro line.\n\n1. **First.** a 1\n2. **Second** b 2\n- **Third**; c 3"
    assert list(cn.split_items(block)) == ["First.", "Second", "Third"]


def test_compare_reports_each_status():
    frozen = {"A": "**A** +276 %", "B": "**B** 0.28 %", "C": "**C** x 1", "D": "**D** 5"}
    regen = {"A": "**A** +276 %", "B": "**B** 0.29 %", "C": "**C** y 1"}
    got = {t: s for t, s, _ in cn.compare(frozen, regen)}
    assert got == {"A": "ok", "B": "mismatch", "C": "text-differs", "D": "unverifiable"}
