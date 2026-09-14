import json

import numpy as np
import pytest

from strike_fit import data


def make_records():
    return [
        {"spin_id": "a", "direction": "cw", "clicks_s": [0.0, 1.1, 2.3], "strike": {"t_s": 5.0, "deflector": 3},
         "legacy": {"angle_bin": 12}, "rotor_clicks_s": None, "rotor_angles_rad": None, "flags": []},
        {"spin_id": "b", "direction": "ccw", "clicks_s": [0.0, 1.0, 2.1, 3.3], "strike": None},
    ]


def test_views_and_assertion(tmp_path):
    ds = data.Dataset(data.Header(n_deflectors=8, annotation_mode="reactive"), make_records())
    cand = ds.candidate_view()
    assert cand[0].s == 1 and cand[1].s == -1
    assert cand[0].strike == data.Strike(5.0, 3) and cand[1].strike is None
    assert not hasattr(cand[0], "legacy")
    data.assert_candidate_clean(cand)
    base = ds.baseline_view()
    assert base[0].legacy == {"angle_bin": 12} and base[1].legacy == {}
    with pytest.raises(AssertionError):
        data.assert_candidate_clean(base)
    with pytest.raises(TypeError):
        data.assert_candidate_clean([dict()])
    # round trip
    p = tmp_path / "d.jsonl"
    ds.write_jsonl(p)
    ds2 = data.Dataset.read_jsonl(p)
    assert ds2.header == ds.header
    assert ds2.records == ds.records
    assert json.loads(p.read_text().splitlines()[0])["annotation_mode"] == "reactive"


@pytest.mark.parametrize("bad", [
    {"clicks_s": [0.1, 1.0]},
    {"clicks_s": [0.0, 1.0, 0.9]},
    {"clicks_s": [0.0]},
    {"direction": "clockwise"},
    {"strike": {"t_s": 3.0, "deflector": 8}},
    {"strike": {"t_s": -1.0, "deflector": 1}},
    {"rotor_clicks_s": [0.0, 1.0, 2.0]},                                        # no angles
    {"rotor_angles_rad": [0.0, 3.1, 6.3]},                                       # no clicks
    {"rotor_clicks_s": [0.0, 1.0, 2.0], "rotor_angles_rad": [0.0, 3.1]},        # length
    {"rotor_clicks_s": [0.0, 1.0, 2.0], "rotor_angles_rad": [0.0, 3.1, 3.1]},   # not increasing
])
def test_validation_rejects(bad):
    rec = dict(make_records()[0])
    rec.update(bad)
    with pytest.raises(ValueError):
        data.Dataset(data.Header(), [rec])


def test_rotor_angles_load_with_clicks():
    rec = dict(make_records()[0])
    rec.update({"rotor_clicks_s": [0.0, 1.0, 2.1, 4.4], "rotor_angles_rad": [0.0, np.pi, 2 * np.pi, 4 * np.pi]})
    spin = data.Dataset(data.Header(), [rec]).candidate_view()[0]
    np.testing.assert_array_equal(spin.rotor_angles, [0.0, np.pi, 2 * np.pi, 4 * np.pi])
    np.testing.assert_array_equal(spin.rotor_clicks, [0.0, 1.0, 2.1, 4.4])


def test_header_validation():
    with pytest.raises(ValueError):
        data.Header(annotation_mode="live")
    h = data.Header.from_dict({"n_deflectors": 8, "annotation_mode": "mechanical", "fps": 60.0,
                               "reference_deflector": 0, "video": "x", "custom": 1})
    assert h.video == "x" and h.extra == {"custom": 1}
    assert data.Header.from_dict(h.to_dict()) == h
