import math

from lib.ListNode import *
from lib.DoublyListNode import *
from collections import defaultdict, deque
from lib.TreeNode import *


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

    # 9
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        str_x = str(x)

        for i in range(len(str_x) // 2):
            if str_x[i] != str_x[len(str_x) - 1 - i]:
                return False
        return True

    # Time complexity: O(n)
    # Space complexity: O(n)

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

    # Time complexity: O(nlogn)
    # Space complexity: O(logN)

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

    # 227
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        num = 0
        stack = []
        pre_op = '+'
        s+='+'
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
        node = root

        while node:
            if p.val > node.val and q.val > node.val:
                node = node.right
            elif p.val < node.val and q.val < node.val:
                node = node.left
            else:
                return node

    # Time Complexity: O(N)
    # Space Complexity: O(1)

    # 236
    def lowestCommonAncestor2(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
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
    # Time Complexity: O(N)
    # Space Complexity: O(N)


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

            print(f"mid: {mid}, node.val: {node.val}, left: {left}, right: {right}, p_found: {self.p_found}, q_found: {self.q_found}")

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
