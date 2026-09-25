"""Keep pre-existing inspection tests from appending synthetic runs to user history."""

from scripts.inspect_dataset import main


def isolated_main(arguments):
    return main([*arguments, "--no-history"])
