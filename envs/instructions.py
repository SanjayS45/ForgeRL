"""
Instruction templates for MetaWorld tasks.
"""

from typing import List, Dict

# Basic reach instructions
REACH_INSTRUCTIONS = [
    "reach the red target",
    "move to the target position",
    "reach toward the goal",
    "touch the target marker",
    "extend arm to the target",
    "reach the blue marker",
    "move gripper to goal position",
    "reach the green point",
    "touch the yellow target",
    "reach the target ahead",
]

# Push instructions
PUSH_INSTRUCTIONS = [
    "push the block to the target",
    "move the object forward",
    "push the cube to the goal",
    "slide the block to the marker",
    "push object to destination",
]

# Pick and place instructions
PICK_PLACE_INSTRUCTIONS = [
    "pick up the object and place it at the target",
    "grab the block and move it to the goal",
    "lift and relocate the object",
    "pick the cube and drop it at the marker",
    "grasp and place the object",
]

# Task-specific instruction mappings
TASK_INSTRUCTIONS: Dict[str, List[str]] = {
    "reach-v3": REACH_INSTRUCTIONS,
    "reach-v2": REACH_INSTRUCTIONS,
    "push-v3": PUSH_INSTRUCTIONS,
    "push-v2": PUSH_INSTRUCTIONS,
    "pick-place-v3": PICK_PLACE_INSTRUCTIONS,
    "pick-place-v2": PICK_PLACE_INSTRUCTIONS,
}

# Default instructions (for backward compatibility)
INSTRUCTIONS = REACH_INSTRUCTIONS


def get_instructions_for_task(task_name: str) -> List[str]:
    """
    Get list of instructions for a specific task.
    
    Args:
        task_name: MetaWorld task name (e.g., 'reach-v3')
        
    Returns:
        List of instruction strings
    """
    return TASK_INSTRUCTIONS.get(task_name, REACH_INSTRUCTIONS)


def sample_instruction(task_name: str = "reach-v3") -> str:
    """
    Sample a random instruction for a task.
    """
    import random
    instructions = get_instructions_for_task(task_name)
    return random.choice(instructions)
