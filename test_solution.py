import pytest
from Solution import Solution
from lib.ListNode import *
from lib.TreeNode import *
from lib.NestedInteger import *

solution = Solution()

def add_newline_before(func):
    def wrapper(*args, **kwargs):
        print()
        return func(*args, **kwargs)
    return wrapper

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

def test_numberToWords():
    num = 123
    output = solution.numberToWords(num)
    print(output)
    assert output == "One Hundred Twenty Three"

def test_mergeAlternately():
    word1 = "abc"
    word2 = "pqr"
    output = solution.mergeAlternately(word1, word2)
    assert output == "apbqcr"

def test_calculate():
    s = "3+2*2"
    output = solution.calculate(s)
    assert output == 7

@add_newline_before
def test_verticalOrder():
    root = list_to_tree([3, 9, 20, None, None, 15, 7])
    print_tree(root)
    output = solution.verticalOrder(root)
    print(output)
    assert output == [[9],[3,15],[20],[7]]

@add_newline_before
def test_isPalindrome():
    x = 121
    output = solution.isPalindrome(x)
    print(output)
    assert output == True

@add_newline_before
def test_lowestCommonAncestor():
    root = list_to_tree([3,5,1,6,2,0,8,None,None,7,4])
    p = root.left
    q = root.right
    print_tree(p)
    print_tree(q)
    output = solution.lowestCommonAncestor(p, q)
    print_tree(output)
    assert output.val == 3


@add_newline_before
def test_lowestCommonAncestor1():
    root = list_to_tree([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5])
    p = root.left
    q = root.right
    print_tree(root)
    output = solution.lowestCommonAncestor1(root, p, q)
    print_tree(output)
    assert output.val == 6

@add_newline_before
def test_lowestCommonAncestor2():
    root = list_to_tree([3,5,1,6,2,0,8,None,None,7,4])
    p = root.left
    q = root.right
    print_tree(root)
    output = solution.lowestCommonAncestor2(root, p, q)
    print_tree(output)
    assert output.val == 3

@add_newline_before
def test_lcaDeepestLeaves():
    root = list_to_tree([3,5,1,6,2,0,8,None,None,7,4])
    print_tree(root)
    output = solution.lcaDeepestLeaves(root)
    print_tree(output)
    assert output.val == 2

@add_newline_before
def test_lowestCommonAncestor3():
    root = list_to_tree([3,5,1,6,2,0,8,None,None,7,4])
    p = root.left
    q = root.right
    print_tree(root)
    output = solution.lowestCommonAncestor3(root, p, q)
    print_tree(output)
    assert output.val == 3

@add_newline_before
def test_lowestCommonAncestor4():
    root = list_to_tree([3,5,1,6,2,0,8,None,None,7,4])
    p = root.left.right.left
    q = root.left.right.right
    print_tree(p)
    print_tree(q)
    output = solution.lowestCommonAncestor4(root, [p, q])
    print_tree(output)
    assert output.val == 2

@add_newline_before
def test_lengthOfLongestSubstring():
    s = "abcabcbb"
    output = solution.lengthOfLongestSubstring(s)
    print(output)
    assert output == 3

@add_newline_before
def test_minMeetingRooms():
    intervals = [[0, 30], [5, 10], [15, 20]]
    output = solution.minMeetingRooms(intervals)
    print(output)
    assert output == 2

@add_newline_before
def test_canAttendMeetings():
    intervals = [[0, 30], [5, 10], [15, 20]]
    output = solution.canAttendMeetings(intervals)
    print(output)
    assert output == False

@add_newline_before
def test_isValid():
    s = "()"
    output = solution.isValid(s)
    assert output == True

@add_newline_before
def test_pickIndex():
    s = Solution.SolutionPickIndex([1])
    output = s.pickIndex()
    print(output)
    assert output == 0

@add_newline_before
def test_spiralMatrixIII():
    rows = 1
    cols = 4
    rStart = 0
    cStart = 0
    output = solution.spiralMatrixIII(rows, cols, rStart, cStart)
    print(output)
    assert output == [[0,0],[0,1],[0,2],[0,3]]

@add_newline_before
def test_depthSum():
    nestedList = list_to_nested_integer([[1, 1], 2, [1, 1]])
    output = solution.depthSum(nestedList)
    print(output)
    assert output == 10

@add_newline_before
def test_largestNumber():
    nums = [10, 2]
    output = solution.largestNumber(nums)
    print(output)
    assert output == "210"

@add_newline_before
def test_validPalindrome():
    s = "aba"
    output = solution.validPalindrome(s)
    print(output)
    assert output == True

@add_newline_before
def test_regionsBySlashes():
    grid = [" /","/ "]
    output = solution.regionsBySlashes(grid)
    print(output)
    assert output == 2

@add_newline_before
def test_search():
    nums = [-1, 0, 3, 5, 9, 12]
    target = 9
    output = solution.search(nums, target)
    print(output)
    assert output == 4
