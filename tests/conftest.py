import pytest

from legible_motion.envs import MultiObjectPickAndPlaceEnv


@pytest.fixture
def env():
    environment = MultiObjectPickAndPlaceEnv()

    try:
        environment.reset(seed=42)
        yield environment
    finally:
        environment.close()
