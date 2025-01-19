from lib.ListNode import *
from lib.DoublyListNode import *


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

    # Time complexity: O(n)
    # Space complexity: O(n)

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

    # 5
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        def expand(i, j):
            left, right = i, j
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1

        res = [0 ,0]
        for i in range(len(s)):
            odd_length = expand(i, i)
            if odd_length > res[1] - res[0] + 1:
                center = odd_length // 2
                res = [i - center, i + center]

            even_length = expand(i, i + 1)
            if even_length > res[1] - res[0] + 1:
                center = (even_length // 2) - 1
                res = [i - center, i + 1 + center]

        print(f"res: {res}")

        return s[res[0]: res[1] + 1]

    # Time complexity: O(n2)
    # Space complexity: O(1)

    # 42
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left, right = 0, len(height) - 1
        ans = 0
        left_max, right_max = 0, 0
        while left < right:
            if height[left] < height[right]:
                left_max = max(left_max, height[left])
                ans += left_max - height[left]
                left += 1
            else:
                right_max = max(right_max, height[right])
                ans += right_max - height[right]
                right -= 1
        return ans

    # Time complexity: O(n)
    # Space complexity: O(1)

    # 56
    def merge2(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        intervals = sorted(intervals)

        ans = []

        for interval in intervals:
            print(f"interval: {interval}, ans: {ans}")
            if not ans or ans[-1][1] < interval[0]:
                ans.append(interval)
            else:
                ans[-1][1] = max(ans[-1][1], interval[1])

        return ans

    # 88
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        p1 = m - 1
        p2 = n - 1
        for p in range(m + n - 1, -1, -1):
            if p2 < 0:
                break
            if p1 >= 0 and nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1

    # Time complexity: O(n + m)
    # Space complexity: O(1)

    # 121
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        min_price = float("inf")
        max_profit = 0
        for i in range(len(prices)):
            if prices[i] < min_price:
                min_price = prices[i]
            elif prices[i] - min_price > max_profit:
                max_profit = prices[i] - min_price

        return max_profit

    # Time complexity: O(n)
    # Space complexity: O(1)


    # 168
    class LRUCache(object):

        def __init__(self, capacity):
            """
            :type capacity: int
            """
            self.capacity = capacity
            self.dic = {}
            self.head = ListNode(-1, -1)
            self.tail = ListNode(-1, -1)
            self.head.next = self.tail
            self.tail.prev = self.head

        def get(self, key):
            """
            :type key: int
            :rtype: int
            """
            if key not in self.dic:
                return -1

            node = self.dic[key]
            self.remove(node)
            self.add(node)
            return node.val

        def put(self, key, value):
            """
            :type key: int
            :type value: int
            :rtype: None
            """
            if key in self.dic:
                old_node = self.dic[key]
                self.remove(old_node)

            node = DoublyListNode(key, value)
            self.dic[key] = node
            self.add(node)

            if len(self.dic) > self.capacity:
                node_to_delete = self.head.next
                self.remove(node_to_delete)
                del self.dic[node_to_delete.key]

        def add(self, node):
            previous_end = self.tail.prev
            previous_end.next = node
            node.prev = previous_end
            node.next = self.tail
            self.tail.prev = node

        def remove(self, node):
            node.prev.next = node.next
            node.next.prev = node.prev

    # 200
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0

        num = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    self.dfs(grid, i, j)
                    num += 1
        return num

    def dfs(self, grid, r, c):
        if r < 0 or r > len(grid) - 1 or c < 0 or c > len(grid[0]) - 1 or grid[r][c] != "1":
            return

        grid[r][c] = "0"
        self.dfs(grid, r - 1, c)
        self.dfs(grid, r + 1, c)
        self.dfs(grid, r, c - 1)
        self.dfs(grid, r, c + 1)

    # Time complexity: O(M×N)
    # Space complexity: O(M×N)

    # 408
    def validWordAbbreviation(self, word, abbr):
        """
        :type word: str
        :type abbr: str
        :rtype: bool
        """
        i, j = 0, 0
        m, n = len(word), len(abbr)
        while i < m and j < n:
            if word[i] == abbr[j]:
                i += 1
                j += 1
            elif abbr[j] == "0":
                return False
            elif abbr[j].isnumeric():
                k = j
                while k < n and abbr[k].isnumeric():
                    k += 1
                num = int(abbr[j:k])
                i += num
                j = k
            else:
                return False
        return i == m and j == n

    # Time complexity: O(n)
    # Space complexity: O(1)

    # 1249
    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str
        """

        res = set()
        stack = []

        for i,c in enumerate(s):
            if c not in "()":
                continue
            if c == "(":
                stack.append(i)
            elif not stack:
                res.add(i)
            else:
                stack.pop()

        res = res.union(set(stack))
        string_builder = []

        for i, c in enumerate(s):
            if i not in res:
                string_builder.append(c)

        return "".join(string_builder)

    # Time complexity: O(n)
    # Space complexity: O(n)

