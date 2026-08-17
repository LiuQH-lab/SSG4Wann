from pathlib import Path
import tempfile
import unittest

from ssg4wann.exceptions import ConfigParseError
from ssg4wann.parsergen.inload import infoload


class ConfigBooleanParsingTests(unittest.TestCase):
    def load_config(self, content: str):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "sg.in"
            config_path.write_text(content, encoding="utf-8")
            return infoload(str(config_path), rank=0)

    def test_supported_boolean_values_are_parsed_case_insensitively(self):
        true_values = ("True", "T", ".TRUE.", "true", "t", ".true.")
        false_values = ("False", "F", ".FALSE.", "false", "f", ".false.")

        for value in true_values:
            with self.subTest(value=value):
                config = self.load_config(
                    f"soc = {value}\n"
                    "NONCOLLINEAR_channel = True\n"
                    "spin_direction = 0 0 1\n"
                )
                self.assertTrue(config.soc)

        for value in false_values:
            with self.subTest(value=value):
                config = self.load_config(
                    f"soc = {value}\n"
                    "NONCOLLINEAR_channel = True\n"
                    "spin_direction = 0 0 1\n"
                )
                self.assertFalse(config.soc)

    def test_invalid_boolean_value_raises_config_parse_error(self):
        invalid_values = ("Flase", "yes", "1", "0", "enabled")

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ConfigParseError,
                    rf"Invalid boolean value for 'soc' at line 1: '{value}'",
                ):
                    self.load_config(
                        f"soc = {value}\nNONCOLLINEAR_channel = True\n"
                    )

    def test_empty_boolean_value_raises_config_parse_error(self):
        with self.assertRaisesRegex(
            ConfigParseError,
            r"Invalid boolean value for 'soc' at line 1: ''",
        ):
            self.load_config("soc =\nNONCOLLINEAR_channel = True\n")

    def test_missing_soc_raises_config_parse_error(self):
        with self.assertRaisesRegex(ConfigParseError, r"soc variable is not set"):
            self.load_config("NONCOLLINEAR_channel = True\n")


if __name__ == "__main__":
    unittest.main()
