from typing import List, Dict

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

PUSH_INSTRUCTIONS = [
    "push the block to the target",
    "move the object forward",
    "push the cube to the goal",
    "slide the block to the marker",
    "push object to destination",
]

PICK_PLACE_INSTRUCTIONS = [
    "pick up the object and place it at the target",
    "grab the block and move it to the goal",
    "lift and relocate the object",
    "pick the cube and drop it at the marker",
    "grasp and place the object",
]

TASK_INSTRUCTIONS: Dict[str, List[str]] = {
    "reach-v3": REACH_INSTRUCTIONS,
    "reach-v2": REACH_INSTRUCTIONS,
    "push-v3": PUSH_INSTRUCTIONS,
    "push-v2": PUSH_INSTRUCTIONS,
    "pick-place-v3": PICK_PLACE_INSTRUCTIONS,
    "pick-place-v2": PICK_PLACE_INSTRUCTIONS,
}

INSTRUCTIONS = REACH_INSTRUCTIONS

def get_instructions_for_task(task_name: str) -> List[str]:
    
    return TASK_INSTRUCTIONS.get(task_name, REACH_INSTRUCTIONS)

def sample_instruction(task_name: str = "reach-v3") -> str:
    
    import random
    instructions = get_instructions_for_task(task_name)
    return random.choice(instructions)
