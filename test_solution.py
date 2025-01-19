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

def test_LRUCache():
    lRUCache = solution.LRUCache(2)

    lRUCache.put(1, 1) # Cache is {1=1}
    lRUCache.put(2, 2) # Cache is {1=1, 2=2}
    assert lRUCache.get(1) == 1 # Returns 1
    lRUCache.put(3, 3) # Evicts key 2, cache is {1=1, 3=3}
    assert lRUCache.get(2) == -1 # Returns -1 (not found)
    lRUCache.put(4, 4) # Evicts key 1, cache is {4=4, 3=3}
    assert lRUCache.get(1) == -1 # Returns -1 (not found)
    assert lRUCache.get(3) == 3 # Returns 3
    assert lRUCache.get(4) == 4 # Returns 4

def test_trap():
    height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    output = solution.trap(height)
    print(output)
    assert output == 6

def test_numIslands():
    grid = [
      ["1","1","1","1","0"],
      ["1","1","0","1","0"],
      ["1","1","0","0","0"],
      ["0","0","0","0","0"]
    ]
    output = solution.numIslands(grid)
    print(output)
    assert output == 1

def test_validWordAbbreviation():
    word = "internationalization"
    abbr = "i5a11o1"
    output = solution.validWordAbbreviation(word, abbr)
    print(output)
    assert output == True

def test_longestPalindrome():
    s = "babad"
    output = solution.longestPalindrome(s)
    print(output)
    assert output == "bab"

def test_minRemoveToMakeValid():
    s = "lee(t(c)o)de)"
    output = solution.minRemoveToMakeValid(s)
    print(output)
    assert output == "lee(t(c)o)de"

def test_maxProfit():
    prices = [7, 1, 5, 3, 6, 4]
    output = solution.maxProfit(prices)
    print(output)
    assert output == 5

def test_merge2():
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    output = solution.merge2(intervals)
    print(output)
    assert output == [[1,6],[8,10],[15,18]]