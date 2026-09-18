"""Versioned gesture semantics for the inspected AndroidWorld backend."""

GESTURE_POLICIES = ('native_v1', 'finger_swipe_v1')
FINGER_SWIPE_INSTRUCTION = (
    '\nFor swipe, direction is the physical movement of your finger: '
    'left moves from right to left, right from left to right, up from bottom '
    'to top, and down from top to bottom. For scroll, direction is the '
    'direction of content you want to reveal; scroll down reveals content below.'
)


def map_android_gesture(action, policy='native_v1'):
    """Translate policy actions without mutating raw/history actions.

    The bound upstream backend reverses horizontal swipe names relative to
    finger movement, while its vertical swipe names already match. Scroll
    directions describe navigation through content and must stay unchanged.
    """
    if policy not in GESTURE_POLICIES:
        raise ValueError('Unknown Android gesture policy')
    result = dict(action)
    if policy == 'finger_swipe_v1' and action.get('action_type') == 'swipe':
        direction = action.get('direction')
        if direction not in ('left', 'right', 'up', 'down'):
            raise ValueError('Swipe requires a valid direction')
        result['direction'] = {'left': 'right', 'right': 'left'}.get(direction, direction)
    return result
