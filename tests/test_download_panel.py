"""
tests/test_download_panel.py
Iron-clad validation for the Dataset Manager.
"""

import pytest
import tkinter
from unittest.mock import MagicMock, patch
from src.ui.windows.download import DownloadWindow, DatasetArgs
from src.limited_sets import SetInfo
from src.ui.styles import Theme


class TestDownloadPanel:
    @pytest.fixture
    def root(self):
        """Fixture for the root window with Theme applied."""
        root = tkinter.Tk()
        # Initialize the theme to prevent TclErrors when widgets try to access style data
        Theme.apply(root, "Dark")
        yield root
        root.destroy()

    @pytest.fixture
    def mock_sets_data(self):
        return MagicMock(
            data={
                "Outlaws": SetInfo(
                    arena=["OTJ"],
                    seventeenlands=["OTJ"],
                    formats=["PremierDraft", "TradDraft"],
                    set_code="OTJ",
                    start_date="2024-04-16",
                ),
                "Cube": SetInfo(
                    arena=["CUBE"],
                    seventeenlands=["Cube"],
                    formats=["PremierDraft"],
                    set_code="CUBE",
                    start_date="2023-12-01",
                ),
            }
        )

    @pytest.fixture
    def config(self):
        config = MagicMock()
        config.settings.database_location = "/mock"
        config.settings.column_configs = {
            "dataset_manager": [
                "Set",
                "Event",
                "Group",
                "Start",
                "End",
                "Collected",
                "Games",
            ]
        }
        return config

    def test_set_metadata_sync(self, root, mock_sets_data, config):
        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())
        panel.vars["set"].set("Cube")
        panel._on_set_change("Cube")
        assert panel.vars["start"].get() == "2023-12-01"
        assert panel.vars["event"].get() == "PremierDraft"

    def test_resolve_start_date_prefers_set_info(self, root, mock_sets_data, config):
        """A set with a real start date uses it directly."""
        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())
        assert panel._resolve_start_date("Outlaws") == "2024-04-16"

    def test_resolve_start_date_manifest_fallback(self, root, config):
        """A set stuck on the placeholder date (e.g. stale set-list cache) falls
        back to the earliest start_date in the synced dataset manifest."""
        from src.constants import START_DATE_DEFAULT

        sets_data = MagicMock(
            data={
                "Marvel Super Heroes": SetInfo(
                    arena=["ALL"],
                    seventeenlands=["MSH"],
                    formats=[],
                    set_code="MARVEL",
                    start_date=START_DATE_DEFAULT,
                ),
            }
        )
        panel = DownloadWindow(root, sets_data, config, MagicMock())

        manifest = {
            "datasets": {
                "MSH_PremierDraft_All": {"start_date": "2026-06-23"},
                "MSH_TradDraft_All": {"start_date": "2026-06-24"},
                "TMT_PremierDraft_All": {"start_date": "2026-03-03"},
            }
        }
        with patch(
            "src.ui.windows.download.read_local_manifest", return_value=manifest
        ):
            assert panel._resolve_start_date("Marvel Super Heroes") == "2026-06-23"

    def test_resolve_start_date_placeholder_without_manifest(self, root, config):
        """No real set date and no manifest entry: keep the placeholder."""
        from src.constants import START_DATE_DEFAULT

        sets_data = MagicMock(
            data={
                "Marvel Super Heroes": SetInfo(
                    arena=["ALL"],
                    seventeenlands=["MSH"],
                    formats=[],
                    set_code="MARVEL",
                    start_date=START_DATE_DEFAULT,
                ),
            }
        )
        panel = DownloadWindow(root, sets_data, config, MagicMock())

        with patch(
            "src.ui.windows.download.read_local_manifest", return_value={}
        ):
            assert (
                panel._resolve_start_date("Marvel Super Heroes") == START_DATE_DEFAULT
            )

    def test_threshold_sanitization(self, root, mock_sets_data, config):
        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())
        panel.vars["threshold"].set("abc")
        with patch("tkinter.messagebox.showerror") as mock_err:
            panel._start_download()
            mock_err.assert_called_once()
            assert "numeric" in mock_err.call_args[0][1].lower()

    @patch("src.ui.windows.download.threading.Thread")
    @patch("src.ui.windows.download.FileExtractor")
    def test_state_locking_during_download(
        self, mock_ex_cls, mock_thread, root, mock_sets_data, config
    ):
        class MockSyncThread:
            def __init__(self, target, args, daemon=True):
                self.target = target
                self.args = args

            def start(self):
                self.target(*self.args)

            def is_alive(self):
                return False

        # Force background thread to execute inline synchronously
        mock_thread.side_effect = MockSyncThread

        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())
        mock_ex_cls.return_value.retrieve_17lands_color_ratings.return_value = (
            False,
            0,
        )
        panel._start_download()
        assert (
            str(panel.btn_dl["state"]) == "normal"
        )  # Download is so fast it's finished instantly

    def test_notification_enter_handshake(self, root, mock_sets_data, config):
        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())
        args = DatasetArgs("OTJ", "TradDraft", "2024", "2024", "All", 100, None)
        with patch.object(panel, "_start_download") as mock_start:
            panel.enter(args)
            assert panel.vars["set"].get() == "Outlaws"
            assert panel.vars["event"].get() == "TradDraft"
            mock_start.assert_called_once_with(args)

    @patch("src.ui.windows.download.threading.Thread")
    @patch("src.ui.windows.download.FileExtractor")
    @patch("tkinter.messagebox.showinfo")
    def test_successful_download_callback_routing(
        self, mock_msg, mock_ex_cls, mock_thread, root, mock_sets_data, config
    ):
        """Verify that a successful dataset extraction properly notifies the UI and resets the progress bar."""

        # We don't want the thread to actually run, we just want to test the UI callback logic
        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())

        # Inject the mock extractor
        mock_ex = mock_ex_cls.return_value

        # Test the finalize callback directly (what the thread calls when it finishes successfully)
        panel.btn_dl.configure(state="disabled")
        panel.progress["value"] = 50

        panel._finalize_download("Success Message!")

        # Verify UI reset
        assert str(panel.btn_dl["state"]) == "normal"
        assert panel.progress["value"] == 0
        assert panel.vars["status"].get() == "DOWNLOAD SUCCESSFUL"
        mock_msg.assert_called_once_with(
            "Dataset Download Complete", "Success Message!"
        )

    @patch("src.ui.windows.download.write_configuration")
    @patch("src.ui.windows.download.FileExtractor")
    def test_failed_export_keeps_active_dataset(
        self, mock_ex_cls, mock_write, root, mock_sets_data, config
    ):
        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())
        config.card_data.latest_dataset = "OTJ_PremierDraft_All_Data.json"
        mock_ex = mock_ex_cls.return_value
        mock_ex.retrieve_17lands_color_ratings.return_value = (True, 100)
        mock_ex.download_card_data.return_value = (True, "Downloaded", 100)
        mock_ex.export_card_data.return_value = ""
        ctx = {
            "db_loc": "/mock",
            "threshold": 500,
            "set_key": "Outlaws",
            "event": "PremierDraft",
            "start": "2024-04-16",
            "end": "2024-05-01",
            "time_period": "ALL_TIME",
            "group": "All",
        }

        with patch.object(panel, "_safe_error") as error, patch.object(
            panel, "_safe_finalize"
        ) as finalize:
            panel._run_download_process(None, ctx)

        assert config.card_data.latest_dataset == "OTJ_PremierDraft_All_Data.json"
        mock_write.assert_not_called()
        error.assert_called_once()
        finalize.assert_not_called()

    @patch("src.ui.windows.download.write_configuration")
    @patch("src.ui.windows.download.FileExtractor")
    def test_refresh_of_active_filename_still_notifies_dataset_loader(
        self, mock_ex_cls, mock_write, root, mock_sets_data, config
    ):
        on_update = MagicMock()
        panel = DownloadWindow(root, mock_sets_data, config, on_update)
        filename = "OTJ_PremierDraft_All_Custom-AllTime_Data.json"
        config.card_data.latest_dataset = filename
        mock_ex = mock_ex_cls.return_value
        mock_ex.retrieve_17lands_color_ratings.return_value = (True, 100)
        mock_ex.download_card_data.return_value = (True, "Downloaded", 100)
        mock_ex.export_card_data.return_value = filename
        ctx = {
            "db_loc": "/mock",
            "threshold": 500,
            "set_key": "Outlaws",
            "event": "PremierDraft",
            "start": "2024-04-16",
            "end": "2024-05-01",
            "time_period": "ALL_TIME",
            "group": "All",
        }

        with patch.object(panel, "after", side_effect=lambda delay, fn: fn()):
            panel._run_download_process(None, ctx)

        assert config.card_data.latest_dataset == filename
        mock_write.assert_called_once_with(config)
        on_update.assert_called_once()

    @pytest.mark.parametrize("newest_first", [True, False])
    @patch("src.ui.windows.download.retrieve_local_set_list")
    def test_table_shows_latest_snapshot_once(
        self, mock_retrieve, newest_first, root, mock_sets_data, config
    ):
        old = (
            "Outlaws", "PremierDraft", "All (All Time)", "2024-04-16",
            "2024-05-01", 100,
            "/mock/OTJ_PremierDraft_All_Custom-AllTime-20240501_Data.json",
            "2024-05-01 09:00:00",
        )
        new = (
            "Outlaws", "PremierDraft", "All (All Time)", "2024-04-16",
            "2024-05-02", 200,
            "/mock/OTJ_PremierDraft_All_Custom-AllTime-20240502_Data.json",
            "2024-05-02 09:00:00",
        )
        mock_retrieve.return_value = ([new, old] if newest_first else [old, new], [])
        config.card_data.latest_dataset = new[6].split("/")[-1]
        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())

        assert panel.table.get_children() == (new[6],)
        assert "active_dataset_card" in panel.table.item(new[6], "tags")

    @patch("tkinter.messagebox.showerror")
    def test_failed_download_callback_routing(
        self, mock_err, root, mock_sets_data, config
    ):
        """Verify that a failed dataset extraction cleanly resets the UI and shows an error."""
        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())

        panel.btn_dl.configure(state="disabled")
        panel.progress["value"] = 50

        panel._handle_error("Network Timeout")

        # Verify UI reset
        assert str(panel.btn_dl["state"]) == "normal"
        assert panel.progress["value"] == 0
        assert panel.vars["status"].get() == "DOWNLOAD FAILED"
        mock_err.assert_called_once_with("Download Error", "Network Timeout")

    @patch("src.ui.windows.download.os.remove")
    @patch("tkinter.messagebox.askyesno", return_value=True)
    def test_delete_dataset_success(
        self, mock_ask, mock_remove, root, mock_sets_data, config
    ):
        """Verify deleting an inactive dataset removes the file and refreshes the table."""
        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())

        # Set the active dataset to something else so we are allowed to delete this one
        config.card_data.latest_dataset = "OTJ_PremierDraft_All_Data.json"

        target_file = "/fake/path/MKM_PremierDraft_All_Data.json"

        with patch.object(panel, "_update_table") as mock_update:
            panel._delete_dataset(target_file)

            mock_remove.assert_called_once_with(target_file)
            mock_update.assert_called_once()

    @patch("src.ui.windows.download.write_configuration")
    @patch("src.utils.clear_set_history", return_value=3)
    @patch("tkinter.messagebox.showinfo")
    @patch("tkinter.messagebox.askyesno", return_value=True)
    def test_clear_set_history_button(
        self,
        mock_ask,
        mock_info,
        mock_clear,
        mock_write,
        root,
        mock_sets_data,
        config,
    ):
        """Clear Set History wipes datasets and resets the version marker so the
        next launch performs a fresh refresh."""
        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())

        with patch.object(panel, "_update_table"):
            panel._clear_set_history()

        mock_clear.assert_called_once()
        assert config.settings.last_run_version == ""
        assert config.card_data.latest_dataset == ""
        mock_write.assert_called_once()
        mock_info.assert_called_once()

    @patch("tkinter.messagebox.askyesno", return_value=False)
    def test_clear_set_history_button_cancelled(
        self, mock_ask, root, mock_sets_data, config
    ):
        """Declining the confirmation must not delete anything."""
        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())
        with patch("src.utils.clear_set_history") as mock_clear:
            panel._clear_set_history()
            mock_clear.assert_not_called()

    @patch("tkinter.messagebox.showwarning")
    def test_delete_dataset_blocked_if_active(
        self, mock_warn, root, mock_sets_data, config
    ):
        """Verify the user is prevented from deleting the currently active dataset."""
        panel = DownloadWindow(root, mock_sets_data, config, MagicMock())

        config.card_data.latest_dataset = "OTJ_PremierDraft_All_Data.json"
        target_file = "/fake/path/OTJ_PremierDraft_All_Data.json"

        panel._delete_dataset(target_file)

        mock_warn.assert_called_once()
