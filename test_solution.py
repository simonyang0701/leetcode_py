import pytest

from Solution import Solution

solution = Solution()


def test_twoSum():
    nums = [2, 7, 11, 15]
    target = 9
    res = solution.twoSum(nums, target)
    print(res)
    assert res == [0, 1]
