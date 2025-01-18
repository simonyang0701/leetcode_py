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

def test_merge():
    nums1 = [1, 2, 3, 0, 0, 0]
    m = 3
    nums2 = [2, 5, 6]
    n = 3
    solution.merge(nums1, m, nums2, n)
    print(nums1)
    assert nums1 == [1,2,2,3,5,6]