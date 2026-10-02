import json
import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from src.dataset import Dataset
from src.constants import DATA_FIELD_NAME


@pytest.fixture
def local_face_database(tmp_path):
    """Small real SQLite fixture with FRA preparation and mana-token rules."""
    raw = tmp_path / "MTGA_Data" / "Downloads" / "Raw"
    raw.mkdir(parents=True)
    path = raw / "Raw_CardDatabase_metadata.mtga"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE Cards (GrpId INTEGER PRIMARY KEY, TitleId INTEGER,
                OldSchoolManaText TEXT, Types TEXT, Colors TEXT, ColorIdentity TEXT,
                Subtypes TEXT, Supertypes TEXT, AbilityIds TEXT, LinkedFaceGrpIds TEXT,
                LinkedFaceType INTEGER, IsPrimaryCard INTEGER, IsToken INTEGER,
                ExpansionCode TEXT, DigitalReleaseSet TEXT, Rarity INTEGER,
                AbilityIdToLinkedTokenGrpId TEXT);
            CREATE TABLE Localizations_enUS (LocId INTEGER, Formatted INTEGER, Loc TEXT);
            CREATE TABLE Enums (Type TEXT, Value INTEGER, LocId INTEGER);
        """)
        rows = [
            (100, 1, "o2oR", "2", "4", "3,4", "", "", "500:50", "101", 19, 1, 0,
             "FRA", "", 2, ""),
            (101, 2, "o(B/R)", "10", "3,4", "3,4", "", "", "501:51", "100", 20, 0, 0,
             "FRA", "", 2, ""),
            (102, 3, "o1oRoGoG", "2", "4,5", "4,5", "", "2", "8:52,600:53", "", 0, 1, 0,
             "FRA", "", 4, "600:103"),
            (103, 4, "", "1", "", "4,5", "", "", "1131:54", "", 0, 0, 1,
             "FRA", "", 2, ""),
            (104, 5, "o2oG", "2", "5", "5", "", "", "1005:55", "", 0, 1, 0,
             "FRA", "", 2, ""),
            (105, 6, "oRoG", "4", "4,5", "4,5", "", "", "700:56", "", 0, 1, 0,
             "FRA", "", 2, ""),
            (106, 7, "o3oGoU", "10", "2,5", "2,5", "", "", "", "", 0, 1, 0,
             "FRA", "", 2, ""),
        ]
        db.executemany("INSERT INTO Cards VALUES (" + ",".join("?" for _ in rows[0]) + ")", rows)
        text = {1: "Hallway Heckler", 2: "Vicious Verse", 3: "Aerid Konstrari",
                4: "Heartwood", 5: "Greenhouse Propagator", 6: "Konstrari Charm",
                7: "Entrust the Spark", 20: "Red", 21: "Green", 22: "Black", 23: "Blue",
                24: "Creature", 25: "Sorcery", 26: "Artifact", 27: "Instant", 28: "Legendary",
                50: "This creature enters prepared.",
                51: "Vicious Verse deals 1 damage to target opponent.", 52: "Flying",
                53: "When Aerid Konstrari enters or dies, create a Heartwood token.",
                54: "{oT}: Add {oR} or {oG}.", 55: "{oT}: Add {oG}.",
                56: "Choose one — \\n•Add {oCoCoC}."}
        db.executemany("INSERT INTO Localizations_enUS VALUES (?, 1, ?)", text.items())
        # A readable variant is preferred where available; many names and rules
        # above deliberately exist only as Formatted=1.
        db.execute("INSERT INTO Localizations_enUS VALUES (53, 0, ?)", (text[53],))
        db.executemany("INSERT INTO Enums VALUES (?, ?, ?)", [
            ("Color", 4, 20), ("Color", 5, 21), ("Color", 3, 22), ("Color", 2, 23),
            ("CardType", 2, 24), ("CardType", 10, 25), ("CardType", 1, 26),
            ("CardType", 4, 27), ("SuperType", 2, 28),
        ])
    return raw.parent.parent, path


def test_open_file_enriches_primary_faces_in_memory_only(local_face_database, tmp_path):
    from src.advisor.mana_base import count_fixing
    from src.utils import Result

    arena_directory, database_file = local_face_database
    ratings_file = tmp_path / "ratings.json"
    ratings_file.write_text(json.dumps({"meta": {"version": 3}, "card_ratings": {
        "100": {"name": "Hallway Heckler", "types": ["Creature", "Sorcery"],
                "colors": ["B", "R"], "mana_cost": "{2}{R}", "cmc": 3,
                "deck_colors": {"GR": {"gihwr": 60.0}}},
        "102": {"name": "Aerid Konstrari", "tags": ["fixing_ramp"]},
        "104": {"name": "Greenhouse Propagator", "tags": ["fixing_ramp"]},
        "105": {"name": "Konstrari Charm", "tags": ["fixing_ramp"]},
        "106": {"name": "Entrust the Spark"},
        **{str(i): {"name": f"Unmatched card {i}"} for i in range(107, 112)},
    }}))
    original_ratings = ratings_file.read_bytes()
    original_database = database_file.read_bytes()
    dataset = Dataset(db_path=str(arena_directory))
    with patch("requests.get", side_effect=AssertionError("metadata must stay local")), \
         patch("requests.post", side_effect=AssertionError("metadata must stay local")):
        assert dataset.open_file(str(ratings_file)) == Result.VALID
        hallway = dataset.get_data_by_id(["100"])[0]
        assert hallway["types"] == ["Creature"]
        assert hallway["colors"] == ["R"]
        assert hallway["oracle_text"] == "This creature enters prepared."
        assert hallway["prepared_spell"]["mana_cost"] == "{B/R}"
        assert hallway["prepared_spell"]["types"] == ["Sorcery"]
        assert hallway["deck_colors"]["RG"]["gihwr"] == 60.0
        aerid, greenhouse, charm, entrust = dataset.get_data_by_id(["102", "104", "105", "106"])
        assert "Legendary" in aerid["types"]
        assert aerid["produced_tokens"][0]["oracle_text"] == "{T}: Add {R} or {G}."
        assert "{C}{C}{C}" in charm["oracle_text"]
        assert entrust["mana_cost"] == "{3}{G}{U}"
        assert entrust["cmc"] == 5
        assert count_fixing([aerid, greenhouse, charm]) == {
            "W": 0, "U": 0, "B": 0, "R": 1, "G": 2,
        }
    assert ratings_file.read_bytes() == original_ratings
    assert database_file.read_bytes() == original_database
    assert not database_file.with_name(database_file.name + "-journal").exists()


def test_late_arena_association_enriches_existing_record_references(local_face_database):
    arena_directory, _ = local_face_database
    card = {"name": "Hallway Heckler", "types": ["Creature", "Sorcery"],
            "colors": ["B", "R"]}
    dataset = Dataset()
    dataset._dataset = {"card_ratings": {"100": card}}
    dataset._name_index[card["name"]] = card
    dataset.db_path = str(arena_directory)
    assert dataset._name_index["Hallway Heckler"] is card
    assert card["colors"] == ["R"]
    assert card["prepared_spell"]["name"] == "Vicious Verse"


def test_metadata_cache_returns_independent_copies(local_face_database):
    from src.file_extractor import load_local_card_metadata

    arena_directory, _ = local_face_database
    first = load_local_card_metadata(arena_directory, ["100"])
    first["100"]["main_face"]["colors"].append("U")
    with patch("src.file_extractor.sqlite3.connect", side_effect=AssertionError("cached read")):
        second = load_local_card_metadata(arena_directory, ["100"])
    assert second["100"]["main_face"]["colors"] == ["R"]


def test_new_download_also_enriches_an_old_temp_cache(local_face_database, tmp_path, monkeypatch):
    from src import constants
    from src.file_extractor import FileExtractor

    arena_directory, database_file = local_face_database
    cached_file = tmp_path / "temp_card_data.json"
    cached_file.write_text(json.dumps({"FRA": {"100": {
        "name": "Hallway Heckler", "types": ["Creature", "Sorcery"],
        "colors": ["B", "R"], "mana_cost": "{2}{R}", "cmc": 3,
    }}}))
    monkeypatch.setattr(constants, "TEMP_CARD_DATA_FILE", str(cached_file))
    extractor = FileExtractor(str(arena_directory), MagicMock(), MagicMock(), MagicMock())
    extractor.selected_sets = MagicMock(arena=["FRA"])
    result, _, _ = extractor._retrieve_local_arena_data(database_file.stat().st_size)
    assert result
    assert extractor.card_dict["100"]["types"] == ["Creature"]
    assert extractor.card_dict["100"]["colors"] == ["R"]
    assert extractor.card_dict["100"]["prepared_spell"]["name"] == "Vicious Verse"


def test_obsolete_database_without_localized_types_does_not_erase_metadata(local_face_database):
    from src.file_extractor import load_local_card_metadata

    arena_directory, database_file = local_face_database
    with sqlite3.connect(database_file) as db:
        db.execute("DELETE FROM Enums WHERE Type = 'CardType'")
    assert load_local_card_metadata(arena_directory, ["100"]) == {}


def test_automatic_discovery_enriches_fra_without_saving_a_database_setting(local_face_database, tmp_path, monkeypatch):
    from src import constants
    from src.file_extractor import discover_local_arena_metadata_directory

    arena_directory, _ = local_face_database
    monkeypatch.setattr("src.file_extractor.sys.platform", constants.PLATFORM_ID_OSX)
    monkeypatch.setattr(constants, "LOCAL_DATA_FOLDER_PATH_OSX", str(arena_directory))
    monkeypatch.setattr(constants, "LOCAL_DATA_FOLDER_PATH_OSX_STEAM", str(tmp_path / "unused"))
    monkeypatch.setattr("src.file_extractor._LOCAL_ARENA_METADATA_DIRECTORY_CACHE", {})
    dataset = Dataset()
    card = {"name": "Hallway Heckler", "types": ["Creature", "Sorcery"],
            "colors": ["B", "R"], "isprimarycard": 1}
    dataset._dataset = {"card_ratings": {"100": card}}
    assert dataset.enrich_local_metadata() == 1
    assert card["colors"] == ["R"]
    assert card["prepared_spell"]["name"] == "Vicious Verse"
    assert dataset.db_path is None
    with patch("src.file_extractor.Path.is_dir", side_effect=AssertionError("cached discovery")):
        assert discover_local_arena_metadata_directory() == str(arena_directory)


def test_local_metadata_requires_matching_name_and_primary_status(local_face_database):
    arena_directory, _ = local_face_database
    dataset = Dataset(db_path=str(arena_directory))
    unrelated = {"name": "Custom card at a reused ID", "colors": ["G"]}
    wrong_face = {"name": "Vicious Verse", "isprimarycard": 1, "types": ["Creature"]}
    dataset._dataset = {"card_ratings": {"100": unrelated, "101": wrong_face}}
    assert dataset.enrich_local_metadata() == 0
    assert unrelated == {"name": "Custom card at a reused ID", "colors": ["G"]}
    assert wrong_face["types"] == ["Creature"]
    assert "main_face" not in wrong_face


def test_automatic_enrichment_does_not_migrate_other_sets(local_face_database, tmp_path, monkeypatch):
    from src import constants

    arena_directory, database_file = local_face_database
    with sqlite3.connect(database_file) as db:
        db.execute("UPDATE Cards SET ExpansionCode='M10'")
    monkeypatch.setattr("src.file_extractor.sys.platform", constants.PLATFORM_ID_OSX)
    monkeypatch.setattr(constants, "LOCAL_DATA_FOLDER_PATH_OSX", str(arena_directory))
    monkeypatch.setattr(constants, "LOCAL_DATA_FOLDER_PATH_OSX_STEAM", str(tmp_path / "unused"))
    monkeypatch.setattr("src.file_extractor._LOCAL_ARENA_METADATA_DIRECTORY_CACHE", {})
    card = {"name": "Aerid Konstrari", "types": ["Creature"], "colors": ["R", "G"]}
    dataset = Dataset()
    dataset._dataset = {"card_ratings": {"102": card}}
    assert dataset.enrich_local_metadata() == 0
    assert "main_face" not in card


def test_resolve_unknown_id():
    dataset = Dataset(retrieve_unknown=True, db_path="/mock/path")

    with patch("os.path.exists", return_value=True):
        with patch("os.listdir", return_value=["Raw_CardDatabase_1.sqlite"]):
            with patch("os.path.getmtime", return_value=1.0):
                with patch("sqlite3.connect") as mock_connect:
                    mock_conn = MagicMock()
                    mock_cursor = MagicMock()
                    mock_connect.return_value = mock_conn
                    mock_conn.cursor.return_value = mock_cursor

                    # Mock finding the table (fetchall) and row (fetchone)
                    mock_cursor.fetchall.return_value = [("Localizations_enUS",)]
                    mock_cursor.fetchone.return_value = ("Lightning Bolt",)

                    result = dataset._resolve_unknown_id("123")
                    assert result == "Lightning Bolt"
                    assert dataset.unknown_id_cache["123"] == "Lightning Bolt"


def test_resolve_unknown_id_no_db_dir():
    """Verify graceful fallback if the MTGA_Data directory doesn't exist."""
    dataset = Dataset(retrieve_unknown=True, db_path="/fake/path")
    with patch("os.path.exists", return_value=False):
        assert dataset._resolve_unknown_id("999") == "999"


@patch("os.path.exists", return_value=True)
@patch("os.listdir")
@patch("os.path.getmtime")
@patch("sqlite3.connect")
def test_resolve_unknown_id_db_corrupted(
    mock_connect, mock_mtime, mock_listdir, mock_exists
):
    """Verify graceful fallback if sqlite throws an OperationalError (e.g. file locked)."""
    dataset = Dataset(retrieve_unknown=True, db_path="/mock/path")
    mock_listdir.return_value = ["Raw_CardDatabase_1.sqlite"]
    mock_connect.side_effect = Exception("database is locked")

    # Should catch exception, log it, and return the raw ID
    assert dataset._resolve_unknown_id("999") == "999"


@patch("src.dataset.Dataset._save_custom_cache")
@patch("src.dataset.Dataset._load_custom_cache")
@patch("requests.post")
def test_get_data_by_id_scryfall_bulk_fallback(mock_post, mock_load, mock_save):
    """Verify that unresolved IDs trigger a bulk Scryfall request that caches results."""
    dataset = Dataset(retrieve_unknown=True, db_path=None)  # No DB, forces Scryfall

    # Mock Scryfall API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {
                "arena_id": 999,
                "name": "New Day 1 Card",
                "type_line": "Creature",
                "colors": ["G"],
                "cmc": 2,
                "mana_cost": "{1}{G}",
            }
        ]
    }
    mock_post.return_value = mock_response

    # Act: Request an ID we have no data for
    result = dataset.get_data_by_id(["999"])

    # Assert
    assert len(result) == 1
    assert result[0][DATA_FIELD_NAME] == "New Day 1 Card"

    # Verify it was saved to the custom fallback cache
    assert "999" in dataset._fallback_ratings
    assert dataset._fallback_ratings["999"][DATA_FIELD_NAME] == "New Day 1 Card"
    mock_save.assert_called_once()


@patch("src.dataset.Dataset._save_custom_cache")
@patch("src.dataset.Dataset._load_custom_cache")
@patch("requests.post")
def test_get_data_by_id_scryfall_api_failure(mock_post, mock_load, mock_save):
    """Verify that if Scryfall API crashes, it injects an empty dummy card to prevent UI KeyErrors."""
    dataset = Dataset(retrieve_unknown=True, db_path=None)
    mock_post.side_effect = Exception("Network Timeout")

    result = dataset.get_data_by_id(["999"])

    assert len(result) == 1
    assert result[0][DATA_FIELD_NAME] == "999"  # Falls back to the string ID
    assert "deck_colors" in result[0]  # Must be initialized safely!
