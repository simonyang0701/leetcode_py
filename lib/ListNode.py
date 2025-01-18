class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_to_listnode(l):
    dummy = ListNode()
    current = dummy
    for number in l:
        current.next = ListNode(number)
        current = current.next
    return dummy.next

def print_listnode(ln):
    if not ln:
        print("Empty ListNode")
        return
    current = ln
    result = []
    while current:
        result.append(str(current.val))
        current = current.next
    print(" -> ".join(result))

def listnode_to_list(ln):
    result = []
    current = ln
    while current:
        result.append(current.val)
        current = current.next
    return result
