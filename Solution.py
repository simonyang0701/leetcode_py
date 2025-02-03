import math
import random

from lib.ListNode import *
from lib.DoublyListNode import *
from collections import defaultdict, deque
from lib.TreeNode import *
from lib.NestedInteger import *
from functools import cmp_to_key


class Solution(object):
    # 1
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # Time complexity: O(n)
        # Space complexity: O(n)
        hashMap = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashMap:
                return [i, hashMap[complement]]
            hashMap[nums[i]] = i

        return []

    # 2
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: Optional[ListNode]
        :type l2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        # Time complexity: O(max(m, n))
        # Space complexity: O(1)
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

    # 3
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        # Time complexity: O(n2)
        # Space complexity: O(min(m,n))
        max_length = 0
        left = 0
        last_seen = {}

        for right, c in enumerate(s):
            if c in last_seen and last_seen[c] >= left:
                left = last_seen[c] + 1

            max_length = max(max_length, right - left + 1)
            last_seen[c] = right
            print(f"c: {c}, left: {left}, last_seen: {last_seen}, max_length: {max_length}")

        return max_length

    # 5
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        # Time complexity: O(n2)
        # Space complexity: O(1)

        def expand(i, j):
            left, right = i, j
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1

        res = [0, 0]
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

    # 9
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        # Time complexity: O(n)
        # Space complexity: O(n)
        str_x = str(x)

        for i in range(len(str_x) // 2):
            if str_x[i] != str_x[len(str_x) - 1 - i]:
                return False
        return True

    # 20
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # Time complexity: O(n)
        # Space complexity: O(n)
        stack = []
        mapping = {")": "(", "}": "{", "]": "["}

        for char in s:
            if char in mapping.values():
                stack.append(char)
            elif char in mapping.keys():
                if not stack or mapping[char] != stack.pop():
                    return False
        if len(stack) <= 0:
            return True
        else:
            return False

    # 42
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # Time complexity: O(n)
        # Space complexity: O(1)
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

    # 56
    def merge2(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        # Time complexity: O(nlogn)
        # Space complexity: O(logN)
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
        # Time complexity: O(n + m)
        # Space complexity: O(1)
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

    # 121
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # Time complexity: O(n)
        # Space complexity: O(1)
        min_price = float("inf")
        max_profit = 0
        for i in range(len(prices)):
            if prices[i] < min_price:
                min_price = prices[i]
            elif prices[i] - min_price > max_profit:
                max_profit = prices[i] - min_price

        return max_profit

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

    # 179
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        # Time Complexity: O(nlogn)
        # Space Complexity: O(n + S)

        def compare(x, y):
            if x + y > y + x:
                return -1
            elif x + y < y + x:
                return 1
            else:
                return 0

        nums_str = list(map(str, nums))
        nums_str.sort(key=cmp_to_key(compare))

        largest_num = ''.join(nums_str).lstrip('0')
        return largest_num or '0'

    # 200
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        # Time complexity: O(M×N)
        # Space complexity: O(M×N)
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

    # 227
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        num = 0
        stack = []
        pre_op = '+'
        s += '+'
        for c in s:
            print(f"c: {c}")
            if c.isdigit():
                num = num * 10 + int(c)
            elif c == ' ':
                continue
            else:
                if pre_op == '+':
                    stack.append(num)
                elif pre_op == '-':
                    stack.append(-num)
                elif pre_op == '*':
                    stack.append(stack.pop() * num)
                    print(f"stack: {stack}")
                elif pre_op == '/':
                    stack.append(math.trunc(stack.pop() / num))
                num = 0
                pre_op = c

        return sum(stack)

    # 235
    def lowestCommonAncestor1(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        # Time Complexity: O(N)
        # Space Complexity: O(1)
        node = root

        while node:
            if p.val > node.val and q.val > node.val:
                node = node.right
            elif p.val < node.val and q.val < node.val:
                node = node.left
            else:
                return node

    # 236
    def lowestCommonAncestor2(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        # Time Complexity: O(N)
        # Space Complexity: O(N)
        self.ans = None

        def recurse_tree(node):
            if not node:
                return False
            left = recurse_tree(node.left)
            right = recurse_tree(node.right)

            if node == p:
                mid = True
            elif node == q:
                mid = True
            else:
                mid = False

            print(f"mid: {mid}, node.val: {node.val}, left: {left}, right: {right}")

            if (mid and left) or (mid and right) or (left and right):
                self.ans = node

            return mid or left or right

        recurse_tree(root)

        return self.ans

    # 252
    def canAttendMeetings(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: bool
        """
        # Time complexity: O(nlogn)
        # Space complexity: O(1)
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                return False
        return True

    # 253
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        # Time Complexity: O(NlogN)
        # Space Complexity: O(N)
        if not intervals:
            return 0

        used_rooms = 0

        start_timings = sorted([i[0] for i in intervals])
        end_timings = sorted(i[1] for i in intervals)
        print(f"start_timings: {start_timings}, end_timings: {end_timings}")

        end_pointer = 0

        for i in range(len(intervals)):
            if start_timings[i] < end_timings[end_pointer]:
                used_rooms += 1
            else:
                end_pointer += 1

        return used_rooms

    # 273
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        if num == 0:
            return "Zero"

        bigString = ["Thousand", "Million", "Billion"]
        result = self.numberToWordsHelper(num % 1000)
        num //= 1000

        for i in range(len(bigString)):
            if num > 0 and num % 1000 > 0:
                result = self.numberToWordsHelper(num % 1000) + bigString[i] + " " + result
            num //= 1000

        return result.strip()

    def numberToWordsHelper(self, num):
        digitString = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
        teenString = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen",
                      "Nineteen"]
        tenString = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]

        result = ""
        if num > 99:
            result += digitString[num // 100] + " Hundred "

        num %= 100
        # handle 10-19
        if num < 20 and num > 9:
            result += teenString[num - 10] + " "
        # handle 20-99
        else:
            if num >= 20:
                result += tenString[num // 10] + " "
            num %= 10
            if num > 0:
                result += digitString[num] + " "

        return result

    # Time Complexity: O(n)
    # Space Complexity: O(1)

    # 314
    def verticalOrder(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[List[int]]
        """
        columnTable = defaultdict(list)
        queue = deque([(root, 0)])

        while queue:
            node, column = queue.popleft()

            if node is not None:
                columnTable[column].append(node.val)

                queue.append((node.left, column - 1))
                queue.append((node.right, column + 1))

        print(f"columnTable: {columnTable}")
        print(f"sorted(columnTable.keys()): {sorted(columnTable.keys())}")

        result = []
        sorted_columns = sorted(columnTable.keys())

        for column in sorted_columns:
            result.append(columnTable[column])

        return result
        # return [columnTable[x] for x in sorted(columnTable.keys())]

    # 339
    def depthSum(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        def dfsdepthSum(nestedList, depth):
            total = 0
            for nested in nestedList:
                if nested.isInteger():
                    total += nested.getInteger() * depth
                else:
                    total += dfsdepthSum(nested.getList(), depth + 1)
            return total

        return dfsdepthSum(nestedList, 1)

    # Time complexity: O(n)
    # Space complexity: O(n)

    # Time Complexity: O(NlogN)
    # Space Complexity: O(N)

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

    # 528
    class SolutionPickIndex(object):

        def __init__(self, w):
            """
            :type w: List[int]
            """
            print(f"w: {w}")
            self.prefix_sums = []
            prefix_sum = 0
            for weight in w:
                prefix_sum += weight
                self.prefix_sums.append(prefix_sum)

            print(f"prefix_sums: {self.prefix_sums}, prefix_sum: {prefix_sum}")
            self.total_sum = prefix_sum

        def pickIndex(self):
            """
            :rtype: int
            """
            target = self.total_sum * random.random()
            for i, prefix_sum in enumerate(self.prefix_sums):
                if target < prefix_sum:
                    return i

    # 680
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        def check_palindrome(s, i, j):
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1

            return True

        i = 0
        j = len(s) - 1
        while i < j:
            if s[i] != s[j]:
                return check_palindrome(s, i, j - 1) or check_palindrome(s, i + 1, j)
            i += 1
            j -= 1

        return True

    # 704
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                return m
            elif nums[m] < target:
                l = m + 1
            else:
                r = m - 1

        return -1

    # Time complexity: O(logn)
    # Space complexity: O(1)

    # 885
    def spiralMatrixIII(self, rows, cols, rStart, cStart):
        """
        :type rows: int
        :type cols: int
        :type rStart: int
        :type cStart: int
        :rtype: List[List[int]]
        """
        dir = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        traversed = []

        # Initial step size is 1, value of d represents the current direction.
        step = 1
        direction = 0
        while len(traversed) < rows * cols:
            # direction = 0 -> East, direction = 1 -> South
            # direction = 2 -> West, direction = 3 -> North
            for _ in range(2):
                for _ in range(step):
                    # Validate the current position
                    if rStart >= 0 and rStart < rows and cStart >= 0 and cStart < cols:
                        traversed.append([rStart, cStart])
                    # Make changes to the current position.
                    rStart += dir[direction][0]
                    cStart += dir[direction][1]

                direction = (direction + 1) % 4
            step += 1
        return traversed

    # Time complexity: O(max(rows, cols)2)
    # Space complexity: O(rows⋅cols)

    # 959
    def regionsBySlashes(self, grid):
        """
        :type grid: List[str]
        :rtype: int
        """
        def dfs(i, j):
            if min(i, j) < 0 or max(i, j) >= len(g) or g[i][j] != 0:
                return 0
            g[i][j] = 1
            return 1 + dfs(i - 1, j) + dfs(i + 1, j) + dfs(i, j - 1) + dfs(i, j + 1)

        n, regions = len(grid), 0
        g = [[0] * n * 3 for i in range(n * 3)]
        for i in range(n):
            for j in range(n):
                if grid[i][j] == '/':
                    g[i * 3][j * 3 + 2] = g[i * 3 + 1][j * 3 + 1] = g[i * 3 + 2][j * 3] = 1
                elif grid[i][j] == '\\':
                    g[i * 3][j * 3] = g[i * 3 + 1][j * 3 + 1] = g[i * 3 + 2][j * 3 + 2] = 1
        for i in range(n * 3):
            for j in range(n * 3):
                regions += 1 if dfs(i, j) > 0 else 0
        return regions

    # 1123
    def lcaDeepestLeaves(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: Optional[TreeNode]
        """

        def height(node):
            if not node:
                return 0
            return max(height(node.left), height(node.right)) + 1

        def dfs(node):
            if not node:
                return None
            left = height(node.left)
            right = height(node.right)

            if left == right:
                return node
            if left > right:
                return dfs(node.left)
            if left < right:
                return dfs(node.right)

        return dfs(root)

    # Time complexity: O(n2)
    # Space complexity: O(n)

    # 1249
    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str
        """

        res = set()
        stack = []

        for i, c in enumerate(s):
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

    # 1644
    def lowestCommonAncestor3(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        self.ans = None
        self.p_found = False
        self.q_found = False

        def recurse_tree(node):
            if not node:
                return False
            left = recurse_tree(node.left)
            right = recurse_tree(node.right)

            if node == p:
                mid = True
                self.p_found = True
            elif node == q:
                mid = True
                self.q_found = True
            else:
                mid = False

            print(
                f"mid: {mid}, node.val: {node.val}, left: {left}, right: {right}, p_found: {self.p_found}, q_found: {self.q_found}")

            if (mid and left) or (mid and right) or (left and right):
                self.ans = node

            return mid or left or right

        recurse_tree(root)

        if self.p_found and self.q_found:
            return self.ans
        else:
            return None

    # Time Complexity: O(N)
    # Space Complexity: O(N)

    # 1650
    def lowestCommonAncestor(self, p, q):
        """
        :type node: Node
        :rtype: Node
        """
        ancestors = set()

        def dfs(node):
            if not node:
                return None
            if node in ancestors:
                return node
            ancestors.add(node)
            return dfs(node.parent)

        a = dfs(p)
        b = dfs(q)
        return a or b

    # Time Complexity: O(log(N))
    # Space Complexity: O(log(N))

    # 1676
    def lowestCommonAncestor4(self, root, nodes):
        """
        :type root: TreeNode
        :type nodes: List[TreeNode]
        """
        nodes = set(nodes)

        def recurse_tree(node):
            if not node:
                return None
            if node in nodes:
                return node

            left = recurse_tree(node.left)
            right = recurse_tree(node.right)

            if left and right:
                return node
            elif left:
                return left
            else:
                return right

        return recurse_tree(root)

    # Time Complexity: O(N)
    # Space Complexity: O(N)

    # 1768
    def mergeAlternately(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: str
        """
        result = []
        n = len(word1) + len(word2) + 1
        for i in range(n):
            if i < len(word1):
                result.append(word1[i])
            if i < len(word2):
                result.append(word2[i])

        return "".join(result)

    # Time complexity: O(m + n)
    # Space complexity: O(1)
