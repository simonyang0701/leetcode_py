import pytest

from Solution import Solution
from lib.Node import *

solution = Solution()

def test_twoSum():
    nums = [2, 7, 11, 15]
    target = 9
    res = solution.twoSum(nums, target)
    print(res)
    assert res == [0, 1]

def test_isPossibleToCutPath():
    grid = [[1, 1, 1], [1, 0, 0], [1, 1, 1]]
    res = solution.isPossibleToCutPath(grid)
    print(res)
    assert res is True

def test_findCheapestPrice():
    n = 4
    flights = [[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]]
    src = 0
    dst = 3
    k = 1
    res = solution.findCheapestPrice(n, flights, src, dst, k)
    print(res)
    assert res == 700

def test_minimumJumps():
    forbidden = [14, 4, 18, 1, 15]
    a = 3
    b = 15
    x = 9
    res = solution.minimumJumps(forbidden, a, b, x)
    print(res)
    assert res == 3

def test_generateParenthesis():
    n = 3
    res = solution.generateParenthesis(n)
    print(res)
    assert res == ["((()))", "(()())", "(())()", "()(())", "()()()"]

def test_merge():
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    res = solution.merge(intervals)
    print(res)
    assert res == [[1,6],[8,10],[15,18]]

def test_insert():
    intervals = [[1,3],[6,9]]
    newInterval = [2,5]
    res = solution.insert(intervals, newInterval)
    print(res)
    assert res == [[1,5],[6,9]]

def test_search():
    nums = [4, 5, 6, 7, 0, 1, 2]
    target = 0
    res = solution.search(nums, target)
    print(res)
    assert res == 4

def test_calculate():
    s = "3+2*2"
    res = solution.calculate(s)
    print(res)
    assert res == 7

def test_lowestCommonAncestor():
    root = init_tree_from_array([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    p = getNode(root, 5)
    q = getNode(root, 1)
    res = solution.lowestCommonAncestor(p,q)
    val = getNode(root, 3)
    print("\n")
    pretty_print(val)
    assert res == val
