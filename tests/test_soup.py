"""The SOUP list and model register: licences and intended use are present for every dependency and model,
the checks fail when they are not, and what is in use (config defaults, the Local AI catalogue) is registered."""

from __future__ import annotations

import json
import os
import sys
from email.message import Message
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import soup_list as soup  # noqa: E402

pytestmark = pytest.mark.req("GC-REQ-048")


def meta(**fields) -> Message:
    m = Message()
    for key, value in fields.items():
        if key == "classifiers":
            for c in value:
                m["Classifier"] = c
        else:
            m[key.replace("_", "-")] = value
    return m


# --- Requirements -------------------------------------------------------------------------------

def test_requirements_are_read_with_extras_comments_and_names_that_start_like_urls(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("# comment\n\nfastapi==0.141.*\nuvicorn[standard]==0.32.*  # inline\nPyJWT[crypto]==2.*\nhttpx==0.27.*\n"
                    "httpie>=3\n-r other.txt\ngit+https://example.org/x.git\nhttps://example.org/x.whl\npython_dotenv==1.*\n  numpy  \n",
                    encoding="utf-8")
    found = soup.read_requirements(path)
    assert [d["name"] for d in found] == ["fastapi", "uvicorn", "PyJWT", "httpx", "httpie", "python_dotenv", "numpy"]
    assert found[1]["extras"] == ["standard"] and found[2]["extras"] == ["crypto"] and found[0]["spec"] == "==0.141.*"


def test_the_real_requirements_file_is_read():
    names = {soup.normalise(d["name"]) for d in soup.read_requirements()}
    assert {"fastapi", "httpx", "pydicom", "pyjwt", "cryptography"} <= names


def test_names_are_normalised():
    assert soup.normalise("PyJWT") == "pyjwt" and soup.normalise("python_dotenv") == "python-dotenv"
    assert soup.normalise("Sentence.Transformers") == "sentence-transformers"


# --- Licences -------------------------------------------------------------------------------------

@pytest.mark.parametrize("fields,want", [
    ({"License_Expression": "MIT"}, "MIT"),
    ({"License": "Apache-2.0"}, "Apache-2.0"),
    ({"License": "UNKNOWN"}, ""),
    ({"License": "A" * 500}, ""),
    ({"License": "Line one\nline two"}, ""),
    ({"classifiers": ["License :: OSI Approved :: MIT License"]}, "MIT"),
    ({"classifiers": ["License :: OSI Approved :: BSD License", "License :: OSI Approved :: Apache Software License"]},
     "BSD OR Apache-2.0"),
    ({"License": "UNKNOWN", "classifiers": ["License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)"]}, "MPL-2.0"),
    ({"classifiers": ["Programming Language :: Python :: 3"]}, ""),
    ({}, ""),
    ({"License_Expression": "BSD-3-Clause", "License": "other"}, "BSD-3-Clause"),
])
def test_licence_from_package_metadata(fields, want):
    assert soup.licence_from_metadata(meta(**fields)) == want


@pytest.mark.parametrize("component,want", [
    ({"licenses": [{"license": {"id": "MIT"}}]}, "MIT"),
    ({"licenses": [{"license": {"name": "Some Custom Licence"}}]}, "Some Custom Licence"),
    ({"licenses": [{"expression": "MIT OR Apache-2.0"}]}, "MIT OR Apache-2.0"),
    ({"licenses": [{"license": {"id": "MIT"}}, {"license": {"id": "BSD-3-Clause"}}]}, "MIT AND BSD-3-Clause"),
    ({"licenses": []}, ""), ({}, ""), ({"licenses": [None, "x", {"license": "str"}, {"license": {}}]}, ""),
])
def test_licence_from_an_sbom_component(component, want):
    assert soup.licence_from_sbom(component) == want


def sbom(tmp_path, components):
    path = tmp_path / "sbom.json"
    path.write_text(json.dumps({"bomFormat": "CycloneDX", "components": components}), encoding="utf-8")
    return path


def test_an_sbom_is_read_by_normalised_name(tmp_path):
    path = sbom(tmp_path, [{"name": "PyJWT", "version": "2.9", "licenses": [{"license": {"id": "MIT"}}]},
                           {"name": "no-version", "licenses": []}, {"version": "1"}, "junk", None])
    found = soup.from_sbom(path)
    assert set(found) == {"pyjwt", "no-version"} and found["pyjwt"]["licence"] == "MIT" and found["no-version"]["version"] == ""


def test_a_bad_sbom_is_an_error(tmp_path):
    for content in ("not json", "[]", "{}", '{"components": "x"}'):
        path = tmp_path / "bad.json"
        path.write_text(content)
        with pytest.raises(soup.CheckError):
            soup.from_sbom(path)
    with pytest.raises(soup.CheckError, match="Can't read"):
        soup.from_sbom(tmp_path / "missing.json")


# --- The list and its failures ------------------------------------------------------------------------

DIRECT = [{"name": "alpha", "spec": "==1.*", "extras": []}, {"name": "Beta_pkg", "spec": "", "extras": []}]
NOTES = {"direct": {"alpha": {"use": "Does a thing.", "failure": "It breaks."}, "beta-pkg": {"use": "Another.", "failure": "x"}},
         "licence_overrides": {}, "reviewed_licences": {}}


def installed(**licences):
    return {k: {"name": k, "version": "1.0", "licence": v, "source": "SBOM"} for k, v in licences.items()}


def test_a_complete_list_has_no_problems():
    result = soup.build_soup(DIRECT, installed(alpha="MIT", **{"beta-pkg": "BSD", "gamma": "Apache-2.0"}), NOTES)
    assert result["problems"] == [] and result["warnings"] == []
    rows = {r["name"]: r for r in result["rows"]}
    assert rows["alpha"]["direct"] and rows["alpha"]["use"] == "Does a thing." and not rows["gamma"]["direct"] and rows["gamma"]["use"] == ""


def test_a_dependency_without_a_licence_fails_and_an_override_fixes_it():
    result = soup.build_soup(DIRECT, installed(alpha="", **{"beta-pkg": "MIT"}), NOTES)
    assert any("alpha: no licence found" in p for p in result["problems"])
    assert [r["licence"] for r in result["rows"] if r["name"] == "alpha"] == ["UNKNOWN"]
    fixed = soup.build_soup(DIRECT, installed(alpha="", **{"beta-pkg": "MIT"}),
                            {**NOTES, "licence_overrides": {"alpha": {"licence": "MIT", "evidence": "its repository"}}})
    assert fixed["problems"] == [] and [r["licence"] for r in fixed["rows"] if r["name"] == "alpha"] == ["MIT"]


def test_a_transitive_dependency_without_a_licence_fails_too():
    result = soup.build_soup(DIRECT, installed(alpha="MIT", **{"beta-pkg": "MIT", "deep": ""}), NOTES)
    assert any("deep: no licence" in p for p in result["problems"])


def test_a_direct_dependency_without_an_intended_use_fails():
    notes = {**NOTES, "direct": {"alpha": {"use": "x"}}}
    assert any("Beta_pkg: a direct dependency with no intended-use" in p
               for p in soup.build_soup(DIRECT, installed(alpha="MIT", **{"beta-pkg": "MIT"}), notes)["problems"])
    for blank in ({"use": ""}, {"use": "   "}, {}, None):
        notes = {**NOTES, "direct": {**NOTES["direct"], "alpha": blank}}
        assert any("alpha: a direct dependency" in p for p in soup.build_soup(DIRECT, installed(alpha="MIT", **{"beta-pkg": "MIT"}), notes)["problems"])


def test_a_requirement_that_is_not_installed_fails():
    result = soup.build_soup(DIRECT, installed(alpha="MIT"), NOTES)
    assert any("Beta_pkg: required but not found" in p for p in result["problems"])


def test_copyleft_licences_warn_until_reviewed():
    for lic in ("GPL-3.0", "AGPL-3.0-or-later", "LGPL-3.0", "MPL-2.0", "EPL-2.0", "GNU General Public License v3 (GPLv3)", "MIT AND MPL-2.0"):
        warned = soup.build_soup(DIRECT, installed(alpha=lic, **{"beta-pkg": "MIT"}), NOTES)
        assert warned["problems"] == [] and any("needs review" in w for w in warned["warnings"]), lic
        ok = soup.build_soup(DIRECT, installed(alpha=lic, **{"beta-pkg": "MIT"}), {**NOTES, "reviewed_licences": {"alpha": "Reviewed: fine."}})
        assert ok["warnings"] == [], lic
    for lic in ("MIT", "BSD-3-Clause", "Apache-2.0", "ISC", "PSF-2.0", "Unlicense"):
        assert soup.build_soup(DIRECT, installed(alpha=lic, **{"beta-pkg": "MIT"}), NOTES)["warnings"] == []


def test_stale_notes_warn():
    notes = {**NOTES, "direct": {**NOTES["direct"], "removed-pkg": {"use": "gone"}}}
    assert any("removed-pkg" in w for w in soup.build_soup(DIRECT, installed(alpha="MIT", **{"beta-pkg": "MIT"}), notes)["warnings"])


def test_the_installed_environment_gives_licences_for_the_real_dependencies():
    direct = soup.read_requirements()
    found = soup.from_environment(direct)
    missing = [d["name"] for d in direct if soup.normalise(d["name"]) not in found]
    if missing:
        pytest.skip(f"not installed here: {', '.join(missing)}")
    assert len(found) >= len(direct)
    assert all(v["version"] for v in found.values())


def test_the_real_soup_list_has_no_problems():
    direct = soup.read_requirements()
    found = soup.from_environment(direct)
    if any(soup.normalise(d["name"]) not in found for d in direct):
        pytest.skip("not every requirement is installed in this environment")
    result = soup.build_soup(direct, found, soup._yaml(soup.SOUP_NOTES))
    assert result["problems"] == [], result["problems"]


def test_every_direct_dependency_in_the_real_notes_says_what_happens_when_it_fails():
    notes = soup._yaml(soup.SOUP_NOTES)
    for name, entry in notes["direct"].items():
        assert entry.get("use", "").strip() and entry.get("failure", "").strip(), name
    direct = {soup.normalise(d["name"]) for d in soup.read_requirements()}
    assert direct <= {soup.normalise(n) for n in notes["direct"]}


# --- Model register -----------------------------------------------------------------------------------

def good_model(**over):
    base = {"id": "m1", "kind": "embedding", "name": "model-a", "version": "1", "provider": "P", "licence": "MIT",
            "intended_use": "Ranks passages.", "not_for": "Deciding.", "data_leaves_machine": False, "notes": "n"}
    return {**base, **over}


def test_a_complete_register_passes():
    assert soup.check_models([good_model()], {"EMBED_MODEL": "model-a"}, []) == []


@pytest.mark.parametrize("field", soup.MODEL_FIELDS + ("data_leaves_machine",))
def test_each_missing_field_is_reported(field):
    model = good_model()
    del model[field]
    problems = soup.check_models([model], {}, [])
    assert problems and any(field in p for p in problems)


@pytest.mark.parametrize("field", soup.MODEL_FIELDS)
def test_blank_or_non_text_fields_are_reported(field):
    for value in ("", "   ", None, 5, ["x"]):
        problems = soup.check_models([good_model(**{field: value})], {}, [])
        assert any(field in p for p in problems), (field, value)


def test_kind_flag_and_duplicates_are_checked():
    assert any("kind must be" in p for p in soup.check_models([good_model(kind="oracle")], {}, []))
    for value in ("false", "yes", 0, None):
        assert any("data_leaves_machine" in p for p in soup.check_models([good_model(data_leaves_machine=value)], {}, []))
    assert any("listed twice" in p for p in soup.check_models([good_model(), good_model()], {}, []))
    assert any("must be a mapping" in p for p in soup.check_models(["x"], {}, []))


def test_a_model_data_leaves_the_machine_for_must_say_its_input_is_de_identified():
    leaky = good_model(kind="generation", data_leaves_machine=True, intended_use="Drafts text.")
    assert any("de-identified" in p for p in soup.check_models([leaky], {}, []))
    assert soup.check_models([{**leaky, "intended_use": "Drafts from a de-identified question."}], {}, []) == []


def test_models_in_use_must_be_registered():
    problems = soup.check_models([good_model()], {"EMBED_MODEL": "model-a", "GEN_MODEL": "other-model", "JUDGE_MODEL": ""}, ["tag:1b"])
    assert any("GEN_MODEL" in p for p in problems) and any("tag:1b" in p for p in problems) and len(problems) == 2


def test_the_real_register_covers_every_default_model_and_catalogue_entry():
    config_models, catalogue = soup.config_and_catalogue()
    assert set(config_models) == {"EMBED_MODEL", "GEN_MODEL", "JUDGE_MODEL"} and len(catalogue) >= 10
    assert soup.check_models(soup.load_models(), config_models, catalogue) == []


def test_changing_a_default_model_without_registering_it_fails(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "GEN_MODEL", "some-new-model")
    config_models, catalogue = soup.config_and_catalogue()
    assert any("some-new-model" in p for p in soup.check_models(soup.load_models(), config_models, catalogue))


def test_the_real_register_states_where_data_goes_and_that_imaging_models_are_not_clinical():
    models = soup.load_models()
    by_id = {m["id"]: m for m in models}
    assert by_id["gen-llama33-70b"]["data_leaves_machine"] is True and by_id["embed-minilm"]["data_leaves_machine"] is False
    assert all(not m["data_leaves_machine"] for m in models if m["id"].startswith("local-"))
    for m in models:
        if m["kind"] == "imaging":
            assert "clinical use" in m["not_for"].lower() and "not a medical device" in m["not_for"].lower()


def test_register_files_that_are_broken_are_errors(tmp_path):
    for content in ("models: []", "models: x", "[]", "just text: [", ""):
        path = tmp_path / "m.yaml"
        path.write_text(content)
        with pytest.raises(soup.CheckError):
            soup.load_models(path)
    with pytest.raises(soup.CheckError, match="Can't read"):
        soup.load_models(tmp_path / "gone.yaml")


def test_models_on_disk_that_are_not_registered_are_reported(tmp_path):
    library = tmp_path / "lib" / "a-model"
    library.mkdir(parents=True)
    (library / "model.json").write_text(json.dumps({"name": "Home grown classifier"}))
    (tmp_path / "lib" / "broken").mkdir()
    (tmp_path / "lib" / "broken" / "model.json").write_text("{not json")
    assert soup.unregistered_library_models([good_model()], tmp_path / "lib") == ["home grown classifier"]
    assert soup.unregistered_library_models([good_model(name="Home grown classifier")], tmp_path / "lib") == []
    assert soup.unregistered_library_models([], tmp_path / "nowhere") == []


# --- The command ------------------------------------------------------------------------------------------------

def run(tmp_path, components, notes=None, extra=()):
    req = tmp_path / "requirements.txt"
    req.write_text("alpha==1.*\n")
    notes_path = tmp_path / "soup.yaml"
    import yaml

    notes_path.write_text(yaml.safe_dump(notes or {"direct": {"alpha": {"use": "x", "failure": "y"}}}))
    return soup.main(["--requirements", str(req), "--sbom", str(sbom(tmp_path, components)), "--notes", str(notes_path),
                      *[str(e) for e in extra]])


def test_the_command_exits_zero_and_writes_all_three_formats(tmp_path, capsys):
    out = {k: tmp_path / f"out.{k}" for k in ("md", "json", "csv")}
    code = run(tmp_path, [{"name": "alpha", "version": "1.2", "licenses": [{"license": {"id": "MIT"}}]},
                          {"name": "dep", "version": "3", "licenses": [{"license": {"id": "BSD-3-Clause"}}]}],
               extra=["--markdown", out["md"], "--json", out["json"], "--csv", out["csv"]])
    assert code == 0
    md = out["md"].read_text()
    assert "| alpha | 1.2 | MIT | x | y |" in md and "| dep | 3 | BSD-3-Clause |" in md and "## Models (" in md
    data = json.loads(out["json"].read_text())
    assert {r["name"] for r in data["soup"]} == {"alpha", "dep"} and data["models"]
    assert out["csv"].read_text().splitlines()[0].startswith("name,version,licence")


def test_the_command_fails_for_a_missing_licence_and_says_which(tmp_path, capsys):
    assert run(tmp_path, [{"name": "alpha", "version": "1"}]) == 1
    assert "alpha: no licence found" in capsys.readouterr().err


def test_the_command_prints_the_markdown_when_no_output_file_is_given(tmp_path, capsys):
    assert run(tmp_path, [{"name": "alpha", "version": "1", "licenses": [{"license": {"id": "MIT"}}]}]) == 0
    assert "# Third-party software" in capsys.readouterr().out


def test_a_markdown_table_cell_cannot_break_the_table():
    result = {"rows": [{"name": "a|b", "version": "1", "licence": "MIT | Apache-2.0", "direct": True, "use": "line\none | two",
                        "failure": "x", "spec": "", "source": "SBOM", "reviewed": ""}], "problems": [], "warnings": []}
    md = soup.to_markdown(result, [good_model()])
    assert "MIT \\| Apache-2.0" in md and "line one \\| two" in md


def test_the_whole_command_passes_on_this_repository(tmp_path):
    direct = soup.read_requirements()
    if any(soup.normalise(d["name"]) not in soup.from_environment(direct) for d in direct):
        pytest.skip("not every requirement is installed in this environment")
    assert soup.main(["--csv", str(tmp_path / "soup.csv")]) == 0
