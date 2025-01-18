import pytest
from Solution import Solution
from lib.ListNode import *

solution = Solution()

def test_twoSum():
    nums = [2, 7, 11, 15]
    target = 9
    output = solution.twoSum(nums, target)
    print(output)
    assert output == [1, 0]

def test_addTwoNumbers():
    l1 = list_to_listnode([2, 4, 3])
    l2 = list_to_listnode([5, 6, 4])
    output = solution.addTwoNumbers(l1, l2)
    print_listnode(output)
    assert listnode_to_list(output) == [7, 0, 8]