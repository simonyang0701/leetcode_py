from collections import deque


class Node:
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.parent = None  # Not necessary for this function


def init_tree_from_array(arr):
    if not arr:  # list is empty
        return None

    root = Node(arr[0])
    node_queue = deque([root])
    idx = 1

    while node_queue and idx < len(arr):
        current_node = node_queue.popleft()

        # Set the left child if the next value in the array isn't None
        if arr[idx] is not None:
            current_node.left = Node(arr[idx])
            current_node.left.parent = current_node  # set the parent pointer
            node_queue.append(current_node.left)

        # Proceed to the next value which should be the right child
        idx += 1

        # Set the right child if the next value in the array isn't None
        if idx < len(arr) and arr[idx] is not None:
            current_node.right = Node(arr[idx])
            current_node.right.parent = current_node  # set the parent pointer
            node_queue.append(current_node.right)

        idx += 1

    return root

def pretty_print(node, level=0):
    if node is not None:
        pretty_print(node.right, level + 1)
        print(' ' * 4 * level + '->', node.val)
        pretty_print(node.left, level + 1)

def getNode(root, value):
    if root is None:
        return None

    if root.val == value:
        return root
    else:
        found_in_left = getNode(root.left, value)
        if found_in_left is not None:
            return found_in_left

        found_in_right = getNode(root.right, value)
        return found_in_right
