from reward import RewardConfig, navigation_reward


def main() -> None:
    config = RewardConfig()
    closer = navigation_reward(
        previous_x=0, previous_y=0, current_x=1, current_y=0,
        target_x=10, target_y=0, goal_radius=0.5, config=config,
    )
    farther = navigation_reward(
        previous_x=1, previous_y=0, current_x=0, current_y=0,
        target_x=10, target_y=0, goal_radius=0.5, config=config,
    )
    stationary = navigation_reward(
        previous_x=0, previous_y=0, current_x=0, current_y=0,
        target_x=10, target_y=0, goal_radius=0.5, config=config,
    )
    goal = navigation_reward(
        previous_x=9, previous_y=0, current_x=9.6, current_y=0,
        target_x=10, target_y=0, goal_radius=0.5, config=config,
    )

    assert closer.reward > 0 and closer.progress == 1
    assert farther.reward < 0 and farther.progress == -1
    assert stationary.reward < 0 and stationary.travelled == 0
    assert goal.reached_goal and goal.reward > config.goal_reward
    print("reward tests passed")


if __name__ == "__main__":
    main()

