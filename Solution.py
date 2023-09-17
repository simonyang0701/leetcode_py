class Solution(object):
    # 1
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        map = {}

        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in map:
                return [map[complement], i]
            map[nums[i]] = i

        return []