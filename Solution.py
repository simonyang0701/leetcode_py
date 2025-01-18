from lib.ListNode import *

class Solution(object):
    # 1
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashMap = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashMap:
                return [i, hashMap[complement]]
            hashMap[nums[i]] = i

        return []
    # Time complexity: O(n).
    # Space complexity: O(n).

    # 2
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: Optional[ListNode]
        :type l2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        dummy = ListNode(0)
        curr = dummy
        carry = 0
        while l1 or l2 or carry != 0:
            l1_val = l1.val if l1 else 0
            l2_val = l2.val if l2 else 0
            sum = l1_val + l2_val + carry
            carry = sum // 10
            div = sum % 10
            curr.next = ListNode(div)
            curr = curr.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        return dummy.next

    # Time complexity: O(max(m, n))
    # Space complexity: O(1)