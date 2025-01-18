import pytest
from Solution import Solution

solution = Solution()

def test_twoSum():
    nums = [2, 7, 11, 15]
    target = 9
    output = solution.twoSum(nums, target)
    print(output)
    assert output == [1, 0]