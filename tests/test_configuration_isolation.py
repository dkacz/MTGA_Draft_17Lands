"""Ensure application tests cannot save over the installed user profile."""

from pathlib import Path

from src import configuration, constants


def test_configuration_defaults_use_disposable_file(isolated_config_path):
    config_path = str(isolated_config_path)
    assert configuration.CONFIG_FILE == config_path
    assert configuration.LOCAL_CONFIG == config_path
    assert Path(configuration.BASE_DIR) == isolated_config_path.parent
    assert Path(constants.BASE_DIR) != isolated_config_path.parent
    for function in (
        configuration.read_configuration,
        configuration.write_configuration,
        configuration.reset_configuration,
    ):
        assert function.__defaults__ == (config_path,)

    original_contents = isolated_config_path.read_bytes()
    try:
        config = configuration.Configuration()
        config.features.override_scale_factor = 1.25
        assert configuration.write_configuration(config)
        saved, success = configuration.read_configuration()
        assert success and saved == config
        assert configuration.reset_configuration()
        reset, success = configuration.read_configuration()
        assert success and reset == configuration.Configuration()
    finally:
        isolated_config_path.write_bytes(original_contents)
