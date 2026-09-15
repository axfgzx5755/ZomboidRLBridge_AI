from actions import Movement
from live_controller import choose_movement
from telemetry import TelemetrySample


def sample(x=0.0, y=0.0):
    return TelemetrySample(x, y, 0.0, 1.0, 1)


def main() -> None:
    assert choose_movement(sample(), 10, 0, 0.1) == Movement.EAST
    assert choose_movement(sample(), -10, 0, 0.1) == Movement.WEST
    assert choose_movement(sample(), 0, -10, 0.1) == Movement.NORTH
    assert choose_movement(sample(), 0, 10, 0.1) == Movement.SOUTH
    assert choose_movement(sample(), 10, -10, 0.1) == Movement.NORTH_EAST
    assert choose_movement(sample(), -10, 10, 0.1) == Movement.SOUTH_WEST
    assert choose_movement(sample(), 0.05, 0.05, 0.1) == Movement.NONE
    print("live controller policy tests passed")


if __name__ == "__main__":
    main()

