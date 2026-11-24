import os
import sys
import time
from pathlib import Path

import pytest

# Run every test in UTC. Python turns plain datetimes into epoch seconds using
# the machine's local time zone, while Spark is set to UTC below; without this,
# a test written as 22:00 is read as 22:00 Hobart time and lands at 11:00 UTC.
os.environ["TZ"] = "UTC"
time.tzset()

# Make `src/` importable when tests run from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def spark():
    """A small local Spark session, shared by all tests.

    The time zone is fixed to UTC so time-window features give the same answer
    on every machine, including the CI runner.
    """
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("fraud-feature-tests")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
