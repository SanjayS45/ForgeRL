METAWORLD_TASKS = {
    "reach-v3": {
        "description": "Reach the red target with the gripper",
        "goal_type": "position",
        "action_dim": 4,
        "obs_dim": 39,
        "max_path_length": 150,
    },
    "push-v3": {
        "description": "Push the block to the target position",
        "goal_type": "object_position",
        "action_dim": 4,
        "obs_dim": 39,
        "max_path_length": 150,
    },
    "pick-place-v3": {
        "description": "Pick up the block and place it in the target area",
        "goal_type": "object_position",
        "action_dim": 4,
        "obs_dim": 39,
        "max_path_length": 150,
    },
    "door-open-v3": {
        "description": "Open the door by rotating its handle",
        "goal_type": "door_angle",
        "action_dim": 4,
        "obs_dim": 39,
        "max_path_length": 150,
    },
    "drawer-open-v3": {
        "description": "Open the drawer by pulling its handle",
        "goal_type": "drawer_position",
        "action_dim": 4,
        "obs_dim": 39,
        "max_path_length": 150,
    },
    "drawer-close-v3": {
        "description": "Close the drawer by pushing its handle",
        "goal_type": "drawer_position",
        "action_dim": 4,
        "obs_dim": 39,
        "max_path_length": 150,
    },
    "button-press-v3": {
        "description": "Press the button",
        "goal_type": "button_state",
        "action_dim": 4,
        "obs_dim": 39,
        "max_path_length": 150,
    },
    "window-open-v3": {
        "description": "Open the window",
        "goal_type": "window_position",
        "action_dim": 4,
        "obs_dim": 39,
        "max_path_length": 150,
    },
    "window-close-v3": {
        "description": "Close the window",
        "goal_type": "window_position",
        "action_dim": 4,
        "obs_dim": 39,
        "max_path_length": 150,
    },
    "faucet-open-v3": {
        "description": "Turn the faucet handle to open it",
        "goal_type": "faucet_state",
        "action_dim": 4,
        "obs_dim": 39,
        "max_path_length": 150,
    },
    "faucet-close-v3": {
        "description": "Turn the faucet handle to close it",
        "goal_type": "faucet_state",
        "action_dim": 4,
        "obs_dim": 39,
        "max_path_length": 150,
    },
}

def get_task_info(task_name: str) -> dict:
    return METAWORLD_TASKS.get(task_name, {})

def get_task_description(task_name: str) -> str:
    info = get_task_info(task_name)
    return info.get("description", f"Complete the {task_name} task")

def list_available_tasks() -> list:
    return list(METAWORLD_TASKS.keys())

