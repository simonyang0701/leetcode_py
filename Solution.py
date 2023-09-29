from collections import defaultdict, deque


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

    # 22
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        answer = []
        def backtracking(cur_string, left_count, right_count):
            if len(cur_string) == 2 * n:
                answer.append("".join(cur_string))
                return
            if left_count < n:
                cur_string.append("(")
                backtracking(cur_string, left_count + 1, right_count)
                cur_string.pop()
            if right_count < left_count:
                cur_string.append(")")
                backtracking(cur_string, left_count, right_count + 1)
                cur_string.pop()
        backtracking([], 0, 0)
        return answer

    # 33
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        n = len(nums)
        left, right = 0, n-1
        while left <= right:
            mid = left + (right - left) // 2

            # Case 1: find target
            if nums[mid] == target:
                return mid

            # Case 2: subarray on mid's left is sorted
            elif nums[mid] >= nums[left]:
                if target >= nums[left] and target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1

            # Case 3: subarray on mid's right is sorted.
            else:
                if target <= nums[right] and target > nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1

    # 56
    # Time complexity: O(nlogn)
    # Space complexity: O(n)
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # if the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged

    # 57
    # Time complexity: O(N)
    # Space complexity: O(1)
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        result = []

        for interval in intervals:
            # the new interval is after the range of other interval, so we can leave the current interval baecause the new one does not overlap with it
            if interval[1] < newInterval[0]:
                result.append(interval)
            # the new interval's range is before the other, so we can add the new interval and update it to the current one
            elif interval[0] > newInterval[1]:
                result.append(newInterval)
                newInterval = interval
            # the new interval is in the range of the other interval, we have an overlap, so we must choose the min for start and max for end of interval
            elif interval[1] >= newInterval[0] or interval[0] <= newInterval[1]:
                newInterval[0] = min(interval[0], newInterval[0])
                newInterval[1] = max(newInterval[1], interval[1])

        result.append(newInterval)
        return result

    # 787
    # Time complexity: O(N+E*K)
    # Space complexity: O(N+E*K)
    def findCheapestPrice(self, n, flights, src, dst, k):
        """
        :type n: int
        :type flights: List[List[int]]
        :type src: int
        :type dst: int
        :type k: int
        :rtype: int
        """
        graph, dis = defaultdict(list), [-1 for _ in range(n)]
        for f, to, price in flights:
            graph[f].append([to, price])

        dis[src], q, step = 0, deque([src]), 0

        while q and step <= k:
            sz = len(q)
            new_dis = list(dis)
            for _ in range(sz):
                cur = q.popleft()
                for neighbor in graph[cur]:
                    if new_dis[neighbor[0]] == -1 or new_dis[neighbor[0]] > dis[cur]+neighbor[1]:
                        new_dis[neighbor[0]] = dis[cur] + neighbor[1]
                        q.append(neighbor[0])
            step += 1
            dis = new_dis

        return dis[dst]

    # 1654
    def minimumJumps(self, forbidden, a, b, x):
        """
        :type forbidden: List[int]
        :type a: int
        :type b: int
        :type x: int
        :rtype: int
        """
        limit = 2000 + a + b
        visited = set(forbidden)
        myque = deque([(0, True)]) # (pos, isForward)
        hops = 0
        while(myque):
            l = len(myque)
            while(l > 0):
                l -= 1
                pos, isForward = myque.popleft()
                if pos == x:
                    return hops
                if pos in visited: continue
                visited.add(pos)
                if isForward:
                    nxt_jump = pos - b
                    if nxt_jump >= 0:
                        myque.append((nxt_jump, False))
                nxt_jump = pos + a
                if nxt_jump <= limit:
                    myque.append((nxt_jump, True))
            hops += 1
        return -1

    # 2556
    # Time complexity: O(n*m)
    # Space complexity: O(n*m)
    def isPossibleToCutPath(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: bool
        """
        m, n = len(grid), len(grid[0])

        #  number of paths from (0, 0) to (i, j)
        dp1 = [[0] * (n + 1) for _ in range(m + 1)]
        dp1[1][1] = 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if grid[i - 1][j - 1]:
                    dp1[i][j] += dp1[i - 1][j] + dp1[i][j - 1]

        #  number of paths from (i, j) to (m-1, n-1)
        dp2 = [[0] * (n + 1) for _ in range(m + 1)]
        dp2[-2][-2] = 1
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if grid[i][j]:
                    dp2[i][j] += dp2[i + 1][j] + dp2[i][j + 1]

        # number of paths from (0, 0) to (m-1, n-1)
        target = dp1[-1][-1]

        for i in range(m):
            for j in range(n):
                if (i != 0 or j != 0) and (i != m - 1 or j != n - 1):
                    if dp1[i + 1][j + 1] * dp2[i][j] == target:
                        return True
        return False
