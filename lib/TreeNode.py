from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None, parent=None):
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent

def list_to_tree(lst):
    if not lst:
        return None

    root = TreeNode(lst[0])
    queue = deque([(root, None, 'root')])
    i = 1

    while queue and i < len(lst):
        node, parent, pos = queue.popleft()

        # Assign parent to current node
        if parent is not None:
            node.parent = parent

        if i < len(lst) and lst[i] is not None:
            node.left = TreeNode(lst[i])
            queue.append((node.left, node, 'left'))
        i += 1

        if i < len(lst) and lst[i] is not None:
            node.right = TreeNode(lst[i])
            queue.append((node.right, node, 'right'))
        i += 1

    return root


def print_tree(root):
    if not root:
        print("Empty tree")
        return

    levels = []

    def traverse(node, depth=0):
        if node:
            if len(levels) == depth:
                levels.append([])
            levels[depth].append(node.val)
            traverse(node.left, depth + 1)
            traverse(node.right, depth + 1)
        else:
            if len(levels) == depth:
                levels.append([])
            levels[depth].append(None)

    traverse(root)

    for depth, level in enumerate(levels):
        indent = " " * (len(levels) - depth - 1) * 4
        print(indent, end="")
        for node in level:
            if node is None:
                print("   ", end="")
            else:
                print(f"{node:2}", end="  ")
        print()

def tree_to_list(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)
    while result and result[-1] is None:
        result.pop()
    return result