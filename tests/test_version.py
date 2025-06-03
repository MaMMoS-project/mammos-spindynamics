import mammos_spindynamics


def test_version():
    """Check that __version__ exists and is a string."""
    assert isinstance(mammos_spindynamics.__version__, str)
