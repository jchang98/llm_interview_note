# LeetCode Hot 100：题型索引、思路与核心代码

> **收录范围**：本页保留来源题单的 91 道题与 Python 面试精简代码；题号/模块以来源为准，部分题目在不同模块重复出现，用于从不同套路复习。

## Fast Look

**知识点链路**：`哈希表 → 双指针/滑动窗口 → 数组与区间 → 链表 → 树 → DP → 单调栈 → 回溯 → 图/BFS/拓扑`。先按数据结构确定状态，再选择遍历、窗口、递推或搜索；复杂度优先看遍历次数、辅助结构和递归深度。

| 模块 | 核心不变量 / 模板 | 高频易错点 |
| --- | --- | --- |
| 哈希表 | value→index、frequency、set 连续起点 | 重复元素与更新顺序 |
| 双指针 | 两端收缩、快慢指针、排序后去重 | 何时移动短板/较小和的一侧 |
| 滑动窗口 | `[left, right]` + 计数器，满足条件后收缩 | 窗口计数减到 0 后的清理与边界 |
| 链表 | dummy、快慢、反转、分治 merge | 断链、尾结点、环入口 |
| 树 | DFS 返回值语义、BFS 队列、BST 范围 | 空节点、左右子树信息、全局答案 |
| 动态规划 | state、transition、base case、遍历顺序 | 初始化、是否可压缩空间、重复子问题 |
| 单调栈 | 栈中元素保持单调，出栈时结算答案 | 哨兵、严格/非严格比较、索引差 |
| 回溯 | 路径、选择列表、结束条件、撤销选择 | 去重、剪枝、原地恢复 |
| 图 | visited / indegree / queue / DFS 状态 | 连通分量、环、层数与原地标记 |

## 题单与核心代码

# 一、哈希表（3题）
## 1. 两数之和
- 思路：哈希表存储`{值:索引}`，遍历查找`target - 当前值`
- 复杂度：**O(n) / O(n)**
```python
def twoSum(nums, target):
    map = {}
    for i, num in enumerate(nums):
        if target - num in map:
            return [map[target-num], i]
        map[num] = i
```

## 49. 字母异位词分组
- 思路：排序字符串作为唯一key，哈希表分组
- 复杂度：**O(nk logk) / O(nk)** （n=字符串数，k=最大字符串长度）
```python
def groupAnagrams(strs):
    from collections import defaultdict
    map = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        map[key].append(s)
    return list(map.values())
```

## 128. 最长连续序列
- 思路：集合去重，找序列起点（num-1不存在），向后统计长度
- 复杂度：**O(n) / O(n)**
```python
def longestConsecutive(nums):
    s = set(nums)
    res = 0
    for num in s:
        if num - 1 not in s:
            cur, cnt = num, 1
            while cur + 1 in s:
                cur += 1
                cnt += 1
            res = max(res, cnt)
    return res
```

---

# 二、双指针（8题）
## 283. 移动零
- 思路：快慢指针，慢指针覆盖非零元素，末尾补零
- 复杂度：**O(n) / O(1)**
```python
def moveZeroes(nums):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1
    for i in range(slow, len(nums)):
        nums[i] = 0
```

## 11. 盛最多水的容器
- 思路：左右指针，**短板向内收缩**求最大面积
- 复杂度：**O(n) / O(1)**
```python
def maxArea(height):
    l, r = 0, len(height)-1
    res = 0
    while l < r:
        area = min(height[l], height[r]) * (r-l)
        res = max(res, area)
        if height[l] < height[r]: l += 1
        else: r -= 1
    return res
```

## 15. 三数之和
- 思路：排序+双指针，跳过重复元素避免重复解
- 复杂度：**O(n²) / O(1)**
```python
def threeSum(nums):
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n):
        if i > 0 and nums[i] == nums[i-1]: continue
        l, r = i+1, n-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l]==nums[l+1]: l+=1
                while l < r and nums[r]==nums[r-1]: r-=1
                l +=1; r -=1
            elif s < 0: l +=1
            else: r -=1
    return res
```

## 42. 接雨水
- 思路：双指针，记录左右最大高度，**短板位置结算雨水量**
- 复杂度：**O(n) / O(1)**
```python
def trap(height):
    l, r = 0, len(height)-1
    l_max = r_max = 0
    res = 0
    while l < r:
        if height[l] < height[r]:
            if height[l] >= l_max: l_max = height[l]
            else: res += l_max - height[l]
            l +=1
        else:
            if height[r] >= r_max: r_max = height[r]
            else: res += r_max - height[r]
            r -=1
    return res
```

## 167. 两数之和 II
- 思路：有序数组，左右指针逼近目标值
- 复杂度：**O(n) / O(1)**
```python
def twoSum(numbers, target):
    l, r = 0, len(numbers)-1
    while l < r:
        s = numbers[l] + numbers[r]
        if s == target: return [l+1, r+1]
        elif s < target: l +=1
        else: r -=1
```

## 344. 反转字符串
- 思路：左右指针交换字符
- 复杂度：**O(n) / O(1)**
```python
def reverseString(s):
    l, r = 0, len(s)-1
    while l < r:
        s[l], s[r] = s[r], s[l]
        l +=1; r -=1
```

## 5. 最长回文子串
- 思路：中心扩散法，枚举奇偶两种回文中心
- 复杂度：**O(n²) / O(1)**
```python
def longestPalindrome(s):
    def expand(l, r):
        while l >=0 and r < len(s) and s[l]==s[r]:
            l -=1; r +=1
        return s[l+1:r]
    res = ''
    for i in range(len(s)):
        s1 = expand(i, i)
        s2 = expand(i, i+1)
        res = max(res, s1, s2, key=len)
    return res
```

## 88. 合并两个有序数组
- 思路：从后往前双指针，避免覆盖元素
- 复杂度：**O(m+n) / O(1)**
```python
def merge(nums1, m, nums2, n):
    i, j, k = m-1, n-1, m+n-1
    while j >=0:
        if i >=0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -=1
        else:
            nums1[k] = nums2[j]
            j -=1
        k -=1
```

---

# 三、滑动窗口（6题）
## 3. 无重复字符的最长子串
- 思路：滑动窗口+哈希表记录字符最后出现位置，收缩左边界
- 复杂度：**O(n) / O(1)**
```python
def lengthOfLongestSubstring(s):
    map = {}
    l = res = 0
    for r, c in enumerate(s):
        if c in map and map[c] >= l:
            l = map[c] + 1
        map[c] = r
        res = max(res, r - l +1)
    return res
```

## 438. 找到所有字母异位词
- 思路：固定长度滑动窗口，字符计数比对
- 复杂度：**O(n) / O(1)**
```python
def findAnagrams(s, p):
    from collections import Counter
    p_cnt = Counter(p)
    s_cnt = Counter(s[:len(p)-1])
    res = []
    for r in range(len(p)-1, len(s)):
        s_cnt[s[r]] +=1
        if s_cnt == p_cnt:
            res.append(r - len(p)+1)
        s_cnt[s[r - len(p)+1]] -=1
    return res
```

## 567. 字符串排列
- 思路：固定窗口+字符计数，判断是否存在异位词
- 复杂度：**O(n) / O(1)**
```python
def checkInclusion(s1, s2):
    from collections import Counter
    cnt1 = Counter(s1)
    cnt2 = Counter(s2[:len(s1)-1])
    for r in range(len(s1)-1, len(s2)):
        cnt2[s2[r]] +=1
        if cnt1 == cnt2: return True
        cnt2[s2[r - len(s1)+1]] -=1
    return False
```

## 76. 最小覆盖子串
- 思路：滑动窗口+计数，满足条件后收缩左边界求最小窗口
- 复杂度：**O(n) / O(1)**
```python
def minWindow(s, t):
    from collections import Counter, defaultdict
    need = Counter(t)
    window = defaultdict(int)
    l, r, match = 0, 0, 0
    start, min_len = 0, float('inf')
    while r < len(s):
        c = s[r]
        if c in need:
            window[c] +=1
            if window[c] == need[c]: match +=1
        r +=1
        while match == len(need):
            if r - l < min_len:
                min_len = r - l
                start = l
            d = s[l]
            if d in need:
                if window[d] == need[d]: match -=1
                window[d] -=1
            l +=1
    return s[start:start+min_len] if min_len != float('inf') else ''
```

## 239. 滑动窗口最大值
- 思路：单调双端队列维护窗口内递减序列，队首为最大值
- 复杂度：**O(n) / O(k)**
```python
def maxSlidingWindow(nums, k):
    from collections import deque
    q = deque()
    res = []
    for i, num in enumerate(nums):
        while q and nums[q[-1]] <= num:
            q.pop()
        q.append(i)
        if q[0] <= i -k:
            q.popleft()
        if i >= k-1:
            res.append(nums[q[0]])
    return res
```

---

# 四、数组（9题）
## 53. 最大子数组和
- 思路：动态规划，当前值=max(自身，自身+前值)
- 复杂度：**O(n) / O(1)**
```python
def maxSubArray(nums):
    cur = res = nums[0]
    for num in nums[1:]:
        cur = max(num, cur + num)
        res = max(res, cur)
    return res
```

## 56. 合并区间
- 思路：排序后，重叠区间合并右端点
- 复杂度：**O(n logn) / O(n)**
```python
def merge(intervals):
    intervals.sort()
    res = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = res[-1]
        if s <= last_e:
            res[-1] = [last_s, max(last_e, e)]
        else:
            res.append([s, e])
    return res
```

## 169. 多数元素
- 思路：摩尔投票法，抵消不同元素
- 复杂度：**O(n) / O(1)**
```python
def majorityElement(nums):
    cnt, cand = 0, 0
    for num in nums:
        if cnt == 0: cand = num
        cnt += 1 if num == cand else -1
    return cand
```

## 189. 轮转数组
- 思路：三次反转数组
- 复杂度：**O(n) / O(1)**
```python
def rotate(nums, k):
    k %= len(nums)
    def reverse(l, r):
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l +=1; r -=1
    reverse(0, len(nums)-1)
    reverse(0, k-1)
    reverse(k, len(nums)-1)
```

## 238. 除自身以外数组的乘积
- 思路：左右乘积列表，分别计算左右侧乘积
- 复杂度：**O(n) / O(1)**
```python
def productExceptSelf(nums):
    n = len(nums)
    res = [1]*n
    left = 1
    for i in range(n):
        res[i] = left
        left *= nums[i]
    right = 1
    for i in range(n-1, -1, -1):
        res[i] *= right
        right *= nums[i]
    return res
```

## 41. 缺失的第一个正数
- 思路：原地哈希，数字归位`nums[i] = i+1`
- 复杂度：**O(n) / O(1)**
```python
def firstMissingPositive(nums):
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i]-1] != nums[i]:
            nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
    for i in range(n):
        if nums[i] != i+1:
            return i+1
    return n+1
```

## 121. 买卖股票的最佳时机
- 思路：记录最低价格，遍历求最大利润
- 复杂度：**O(n) / O(1)**
```python
def maxProfit(prices):
    min_p = float('inf')
    res = 0
    for p in prices:
        min_p = min(min_p, p)
        res = max(res, p - min_p)
    return res
```

## 581. 最短无序连续子数组
- 思路：找无序区间左右边界，确定最小最大
- 复杂度：**O(n) / O(1)**
```python
def findUnsortedSubarray(nums):
    n = len(nums)
    l, r = 0, n-1
    while l < n-1 and nums[l] <= nums[l+1]: l +=1
    if l == n-1: return 0
    while r > 0 and nums[r] >= nums[r-1]: r -=1
    sub = nums[l:r+1]
    min_s, max_s = min(sub), max(sub)
    while l > 0 and nums[l-1] > min_s: l -=1
    while r < n-1 and nums[r+1] < max_s: r +=1
    return r - l +1
```

## 287. 寻找重复数
- 思路：快慢指针（ Floyd判圈）
- 复杂度：**O(n) / O(1)**
```python
def findDuplicate(nums):
    slow = fast = 0
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast: break
    slow = 0
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow
```

---

# 五、链表（10题）
## 206. 反转链表
- 思路：迭代反转指针指向
- 复杂度：**O(n) / O(1)**
```python
def reverseList(head):
    pre, cur = None, head
    while cur:
        nxt = cur.next
        cur.next = pre
        pre = cur
        cur = nxt
    return pre
```

## 21. 合并两个有序链表
- 思路：虚拟头节点，双指针有序合并
- 复杂度：**O(m+n) / O(1)**
```python
def mergeTwoLists(l1, l2):
    dummy = cur = ListNode()
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next
```

## 19. 删除倒数第 N 个节点
- 思路：快慢指针，快指针先走n步
- 复杂度：**O(n) / O(1)**
```python
def removeNthFromEnd(head, n):
    dummy = ListNode(next=head)
    fast = slow = dummy
    for _ in range(n): fast = fast.next
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next
```

## 141. 环形链表
- 思路：快慢指针，相遇则有环
- 复杂度：**O(n) / O(1)**
```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast: return True
    return False
```

## 142. 环形链表 II
- 思路：快慢指针+数学推导找入环点
- 复杂度：**O(n) / O(1)**
```python
def detectCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None
```

## 234. 回文链表
- 思路：快慢指针找中点，反转后半段比对
- 复杂度：**O(n) / O(1)**
```python
def isPalindrome(head):
    pre = None
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        nxt = slow.next
        slow.next = pre
        pre = slow
        slow = nxt
    if fast: slow = slow.next
    while pre and slow:
        if pre.val != slow.val: return False
        pre = pre.next
        slow = slow.next
    return True
```

## 2. 两数相加
- 思路：链表遍历，逐位相加处理进位
- 复杂度：**O(max(m,n)) / O(1)**
```python
def addTwoNumbers(l1, l2):
    dummy = cur = ListNode()
    carry = 0
    while l1 or l2 or carry:
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0
        carry, val = divmod(v1+v2+carry, 10)
        cur.next = ListNode(val)
        cur = cur.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next
```

## 148. 排序链表
- 思路：归并排序，快慢指针分治
- 复杂度：**O(n logn) / O(1)**
```python
def sortList(head):
    if not head or not head.next: return head
    pre = slow = fast = head
    while fast and fast.next:
        pre = slow
        slow = slow.next
        fast = fast.next.next
    pre.next = None
    return merge(sortList(head), sortList(slow))

def merge(l1, l2):
    dummy = cur = ListNode()
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next
```

## 25. K 个一组翻转链表
- 思路：分组反转，递归/迭代衔接
- 复杂度：**O(n) / O(1)**
```python
def reverseKGroup(head, k):
    dummy = ListNode(next=head)
    pre = end = dummy
    while end.next:
        for _ in range(k):
            end = end.next
            if not end: return dummy.next
        start = pre.next
        nxt = end.next
        end.next = None
        pre.next = reverse(start)
        start.next = nxt
        pre = start
        end = pre
    return dummy.next

def reverse(head):
    pre = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = pre
        pre = cur
        cur = nxt
    return pre
```

## 138. 复制带随机指针的链表
- 思路：原地复制+拆分链表
- 复杂度：**O(n) / O(1)**
```python
def copyRandomList(head):
    if not head: return None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = Node(cur.val)
        cur.next.next = nxt
        cur = nxt
    cur = head
    while cur:
        if cur.random:
            cur.next.random = cur.random.next
        cur = cur.next.next
    cur = head
    copy = res = head.next
    while copy.next:
        cur.next = copy.next
        cur = cur.next
        copy.next = cur.next
        copy = copy.next
    cur.next = None
    return res
```

---

# 六、二叉树（15题）
## 104. 最大深度
- 思路：递归，左右子树最大深度+1
- 复杂度：**O(n) / O(h)**
```python
def maxDepth(root):
    if not root: return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

## 226. 翻转二叉树
- 思路：递归交换左右子树
- 复杂度：**O(n) / O(h)**
```python
def invertTree(root):
    if not root: return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root
```

## 102. 层序遍历
- 思路：队列迭代，按层收集节点
- 复杂度：**O(n) / O(n)**
```python
def levelOrder(root):
    if not root: return []
    from collections import deque
    q = deque([root])
    res = []
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level)
    return res
```

## 236. 最近公共祖先
- 思路：递归查找，左右子树均找到则当前节点为祖先
- 复杂度：**O(n) / O(h)**
```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right: return root
    return left or right
```

## 94. 二叉树的中序遍历
- 思路：迭代/递归中序遍历
- 复杂度：**O(n) / O(h)**
```python
def inorderTraversal(root):
    res = []
    def dfs(node):
        if node:
            dfs(node.left)
            res.append(node.val)
            dfs(node.right)
    dfs(root)
    return res
```

## 101. 对称二叉树
- 思路：递归比对左右镜像节点
- 复杂度：**O(n) / O(h)**
```python
def isSymmetric(root):
    def dfs(l, r):
        if not l and not r: return True
        if not l or not r: return False
        return l.val == r.val and dfs(l.left, r.right) and dfs(l.right, r.left)
    return dfs(root.left, root.right)
```

## 105. 从前序与中序遍历序列构造二叉树
- 思路：前序找根，中序分左右子树，递归构建
- 复杂度：**O(n) / O(n)**
```python
def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root = TreeNode(preorder[0])
    idx = inorder.index(preorder[0])
    root.left = buildTree(preorder[1:idx+1], inorder[:idx])
    root.right = buildTree(preorder[idx+1:], inorder[idx+1:])
    return root
```

## 98. 验证二叉搜索树
- 思路：递归限定节点值上下界
- 复杂度：**O(n) / O(h)**
```python
def isValidBST(root):
    def dfs(node, low, high):
        if not node: return True
        if not (low < node.val < high): return False
        return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)
    return dfs(root, float('-inf'), float('inf'))
```

## 114. 二叉树展开为链表
- 思路：后序遍历，拉平右子树衔接
- 复杂度：**O(n) / O(h)**
```python
def flatten(root):
    if not root: return
    flatten(root.left)
    flatten(root.right)
    left = root.left
    right = root.right
    root.left = None
    root.right = left
    cur = root
    while cur.right:
        cur = cur.right
    cur.right = right
```

## 199. 二叉树的右视图
- 思路：层序遍历，取每层最右节点
- 复杂度：**O(n) / O(n)**
```python
def rightSideView(root):
    if not root: return []
    from collections import deque
    q = deque([root])
    res = []
    while q:
        n = len(q)
        for i in range(n):
            node = q.popleft()
            if i == n-1:
                res.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
    return res
```

## 108. 将有序数组转换为二叉搜索树
- 思路：二分取中点为根，递归建树
- 复杂度：**O(n) / O(logn)**
```python
def sortedArrayToBST(nums):
    def build(l, r):
        if l > r: return None
        mid = (l + r) // 2
        root = TreeNode(nums[mid])
        root.left = build(l, mid-1)
        root.right = build(mid+1, r)
        return root
    return build(0, len(nums)-1)
```

## 124. 二叉树中的最大路径和
- 思路：递归计算单链最大和，更新全局最大值
- 复杂度：**O(n) / O(h)**
```python
def maxPathSum(root):
    res = float('-inf')
    def dfs(node):
        nonlocal res
        if not node: return 0
        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)
        res = max(res, node.val + left + right)
        return node.val + max(left, right)
    dfs(root)
    return res
```

## 543. 二叉树的直径
- 思路：递归求左右子树深度和，更新最大直径
- 复杂度：**O(n) / O(h)**
```python
def diameterOfBinaryTree(root):
    res = 0
    def dfs(node):
        nonlocal res
        if not node: return 0
        left = dfs(node.left)
        right = dfs(node.right)
        res = max(res, left + right)
        return max(left, right) + 1
    dfs(root)
    return res
```

## 437. 路径总和 III
- 思路：前缀和+哈希表统计路径数
- 复杂度：**O(n) / O(n)**
```python
def pathSum(root, targetSum):
    from collections import defaultdict
    cnt = defaultdict(int)
    cnt[0] = 1
    def dfs(node, cur):
        if not node: return 0
        cur += node.val
        res = cnt[cur - targetSum]
        cnt[cur] += 1
        res += dfs(node.left, cur) + dfs(node.right, cur)
        cnt[cur] -= 1
        return res
    return dfs(root, 0)
```

## 297. 二叉树的序列化与反序列化
- 思路：前序遍历序列化，递归反序列化
- 复杂度：**O(n) / O(n)**
```python
def serialize(root):
    if not root: return '#'
    return str(root.val) + ',' + serialize(root.left) + ',' + serialize(root.right)

def deserialize(data):
    from collections import deque
    q = deque(data.split(','))
    def dfs():
        val = q.popleft()
        if val == '#': return None
        node = TreeNode(int(val))
        node.left = dfs()
        node.right = dfs()
        return node
    return dfs()
```

---

# 七、动态规划（18题）
## 70. 爬楼梯
- 思路：递推公式`dp[i] = dp[i-1] + dp[i-2]`
- 复杂度：**O(n) / O(1)**
```python
def climbStairs(n):
    a, b = 1, 1
    for _ in range(n-1):
        a, b = b, a+b
    return b
```

## 198. 打家劫舍
- 思路：dp[i] = max(不偷当前，偷当前+前前)
- 复杂度：**O(n) / O(1)**
```python
def rob(nums):
    pre = cur = 0
    for num in nums:
        pre, cur = cur, max(cur, pre + num)
    return cur
```

## 300. 最长递增子序列
- 思路：dp[i] = max(dp[j]+1) （j<i且nums[j]<nums[i]）
- 复杂度：**O(n²) / O(n)**
```python
def lengthOfLIS(nums):
    dp = [1]*len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp)
```

## 1143. 最长公共子序列
- 思路：二维dp，字符相等则对角线+1，否则取左/上最大值
- 复杂度：**O(mn) / O(mn)**
```python
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] +1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

## 62. 不同路径
- 思路：二维dp，边界为1，内部=上+左
- 复杂度：**O(mn) / O(mn)**
```python
def uniquePaths(m, n):
    dp = [[1]*n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]
```

## 64. 最小路径和
- 思路：dp[i][j] = 当前值 + min(上，左)
- 复杂度：**O(mn) / O(mn)**
```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    for i in range(1, m):
        grid[i][0] += grid[i-1][0]
    for j in range(1, n):
        grid[0][j] += grid[0][j-1]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    return grid[-1][-1]
```

## 5. 最长回文子串
- 思路：dp[i][j]表示区间回文，长度分奇偶
- 复杂度：**O(n²) / O(n²)**
```python
def longestPalindrome(s):
    n = len(s)
    dp = [[False]*n for _ in range(n)]
    res = ''
    for l in range(n):
        for i in range(n-l):
            j = i + l
            if l == 0: dp[i][j] = True
            elif l == 1: dp[i][j] = (s[i]==s[j])
            else: dp[i][j] = (s[i]==s[j] and dp[i+1][j-1])
            if dp[i][j] and l+1>len(res):
                res = s[i:j+1]
    return res
```

## 152. 乘积最大子数组
- 思路：记录最大/最小值，负数交换最值
- 复杂度：**O(n) / O(1)**
```python
def maxProduct(nums):
    res = max_p = min_p = nums[0]
    for num in nums[1:]:
        if num < 0:
            max_p, min_p = min_p, max_p
        max_p = max(num, max_p*num)
        min_p = min(num, min_p*num)
        res = max(res, max_p)
    return res
```

## 221. 最大正方形
- 思路：dp[i][j] = min(左，上，左上)+1
- 复杂度：**O(mn) / O(mn)**
```python
def maximalSquare(matrix):
    if not matrix: return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0]*(n+1) for _ in range(m+1)]
    max_s = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            if matrix[i-1][j-1] == '1':
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_s = max(max_s, dp[i][j])
    return max_s **2
```

## 322. 零钱兑换
- 思路：完全背包，dp[i] = min(dp[i-coins[j]]+1)
- 复杂度：**O(amount*n) / O(amount)**
```python
def coinChange(coins, amount):
    dp = [float('inf')]*(amount+1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount+1):
            dp[i] = min(dp[i], dp[i-coin]+1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

## 139. 单词拆分
- 思路：dp[i]表示前i个字符可拆分
- 复杂度：**O(n²) / O(n)**
```python
def wordBreak(s, wordDict):
    wordSet = set(wordDict)
    n = len(s)
    dp = [False]*(n+1)
    dp[0] = True
    for i in range(1, n+1):
        for j in range(i):
            if dp[j] and s[j:i] in wordSet:
                dp[i] = True
                break
    return dp[n]
```

## 309. 最佳买卖股票时机含冷冻期
- 思路：三维状态dp（持有/不持有/冷冻）
- 复杂度：**O(n) / O(1)**
```python
def maxProfit(prices):
    if not prices: return 0
    buy = -prices[0]
    sell = freeze = 0
    for p in prices[1:]:
        new_buy = max(buy, freeze - p)
        new_sell = buy + p
        new_freeze = max(sell, freeze)
        buy, sell, freeze = new_buy, new_sell, new_freeze
    return max(sell, freeze)
```

## 121. 买卖股票的最佳时机
- 思路：记录最低价格，遍历求最大利润
- 复杂度：**O(n) / O(1)**
```python
def maxProfit(prices):
    min_p = float('inf')
    res = 0
    for p in prices:
        min_p = min(min_p, p)
        res = max(res, p - min_p)
    return res
```

## 122. 买卖股票的最佳时机 II
- 思路：贪心，累加所有上涨收益
- 复杂度：**O(n) / O(1)**
```python
def maxProfit(prices):
    res = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            res += prices[i] - prices[i-1]
    return res
```

## 416. 分割等和子集
- 思路：01背包，判断能否凑出总和一半
- 复杂度：**O(n*sum) / O(sum)**
```python
def canPartition(nums):
    total = sum(nums)
    if total % 2 != 0: return False
    target = total // 2
    dp = [False]*(target+1)
    dp[0] = True
    for num in nums:
        for i in range(target, num-1, -1):
            dp[i] = dp[i] or dp[i-num]
    return dp[target]
```

## 494. 目标和
- 思路：01背包，正负号转化为子集和
- 复杂度：**O(n*sum) / O(sum)**
```python
def findTargetSumWays(nums, target):
    total = sum(nums)
    if (total + target) % 2 != 0 or total < abs(target): return 0
    new_t = (total + target) // 2
    dp = [0]*(new_t+1)
    dp[0] = 1
    for num in nums:
        for i in range(new_t, num-1, -1):
            dp[i] += dp[i-num]
    return dp[new_t]
```

## 279. 完全平方数
- 思路：完全背包，求最小数量
- 复杂度：**O(n√n) / O(n)**
```python
def numSquares(n):
    dp = [float('inf')]*(n+1)
    dp[0] = 0
    for i in range(1, n+1):
        j = 1
        while j*j <= i:
            dp[i] = min(dp[i], dp[i-j*j]+1)
            j +=1
    return dp[n]
```

## 96. 不同的二叉搜索树
- 思路：卡特兰数，dp[i] = 累加dp[j]*dp[i-j-1]
- 复杂度：**O(n²) / O(n)**
```python
def numTrees(n):
    dp = [0]*(n+1)
    dp[0] = dp[1] = 1
    for i in range(2, n+1):
        for j in range(i):
            dp[i] += dp[j] * dp[i-j-1]
    return dp[n]
```

---

# 八、栈 / 单调栈（8题）
## 20. 有效的括号
- 思路：栈匹配，右括号匹配栈顶左括号
- 复杂度：**O(n) / O(n)**
```python
def isValid(s):
    stack = []
    map = {')':'(', '}':'{', ']':'['}
    for c in s:
        if c not in map:
            stack.append(c)
        else:
            if not stack or stack.pop() != map[c]:
                return False
    return not stack
```

## 155. 最小栈
- 思路：辅助栈同步存储最小值
- 复杂度：**O(1) / O(n)**
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]
```

## 84. 柱状图最大矩形
- 思路：单调递增栈，记录左右第一个更小元素
- 复杂度：**O(n) / O(n)**
```python
def largestRectangleArea(heights):
    stack = [-1]
    res = 0
    heights.append(0)
    for i, h in enumerate(heights):
        while heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i - stack[-1] -1
            res = max(res, height * width)
        stack.append(i)
    return res
```

## 739. 每日温度
- 思路：单调栈存索引，找下一个更大元素
- 复杂度：**O(n) / O(n)**
```python
def dailyTemperatures(temperatures):
    n = len(temperatures)
    res = [0]*n
    stack = []
    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            idx = stack.pop()
            res[idx] = i - idx
        stack.append(i)
    return res
```

## 402. 移掉K位数字
- 思路：单调递增栈，移除高位大数
- 复杂度：**O(n) / O(n)**
```python
def removeKdigits(num, k):
    stack = []
    for c in num:
        while k > 0 and stack and stack[-1] > c:
            stack.pop()
            k -=1
        stack.append(c)
    while k > 0:
        stack.pop()
        k -=1
    res = ''.join(stack).lstrip('0')
    return res if res else '0'
```

## 316. 去除重复字母
- 思路：单调栈+哈希，保证字典序最小
- 复杂度：**O(n) / O(n)**
```python
def removeDuplicateLetters(s):
    stack = []
    in_stack = set()
    last_idx = {c:i for i,c in enumerate(s)}
    for i,c in enumerate(s):
        if c in in_stack: continue
        while stack and c < stack[-1] and last_idx[stack[-1]] > i:
            in_stack.remove(stack.pop())
        stack.append(c)
        in_stack.add(c)
    return ''.join(stack)
```

## 503. 下一个更大元素 II
- 思路：循环数组，单调栈遍历两次
- 复杂度：**O(n) / O(n)**
```python
def nextGreaterElements(nums):
    n = len(nums)
    res = [-1]*n
    stack = []
    for i in range(2*n):
        num = nums[i%n]
        while stack and num > nums[stack[-1]]:
            idx = stack.pop()
            res[idx] = num
        if i < n:
            stack.append(i)
    return res
```

## 85. 最大矩形
- 思路：转化为柱状图，复用84题解法
- 复杂度：**O(mn) / O(n)**
```python
def maximalRectangle(matrix):
    if not matrix: return 0
    m, n = len(matrix), len(matrix[0])
    heights = [0]*n
    res = 0
    def largest(heights):
        stack = [-1]
        max_a = 0
        heights.append(0)
        for i,h in enumerate(heights):
            while heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i - stack[-1] -1
                max_a = max(max_a, height*width)
            stack.append(i)
        return max_a
    for i in range(m):
        for j in range(n):
            heights[j] = heights[j]+1 if matrix[i][j] == '1' else 0
        res = max(res, largest(heights))
    return res
```

---

# 九、回溯（7题）
## 46. 全排列
- 思路：回溯+used数组，枚举所有排列
- 复杂度：**O(n*n!) / O(n)**
```python
def permute(nums):
    res = []
    def backtrack(path, used):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]: continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False]*len(nums))
    return res
```

## 22. 括号生成
- 思路：回溯，左括号<n、右括号<左括号时添加
- 复杂度：**O(4ⁿ/√n) / O(n)**
```python
def generateParenthesis(n):
    res = []
    def backtrack(s, left, right):
        if len(s) == 2*n:
            res.append(s)
            return
        if left < n:
            backtrack(s+'(', left+1, right)
        if right < left:
            backtrack(s+')', left, right+1)
    backtrack('', 0, 0)
    return res
```

## 78. 子集
- 思路：回溯，无结束条件，直接收集所有路径
- 复杂度：**O(2ⁿ) / O(n)**
```python
def subsets(nums):
    res = []
    def backtrack(path, start):
        res.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(path, i+1)
            path.pop()
    backtrack([], 0)
    return res
```

## 39. 组合总和
- 思路：回溯，可重复选，剪枝
- 复杂度：**O(2ⁿ) / O(n)**
```python
def combinationSum(candidates, target):
    res = []
    def backtrack(path, start, remain):
        if remain == 0:
            res.append(path[:])
            return
        if remain < 0: return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(path, i, remain - candidates[i])
            path.pop()
    backtrack([], 0, target)
    return res
```

## 47. 全排列 II
- 思路：排序去重+回溯
- 复杂度：**O(n*n!) / O(n)**
```python
def permuteUnique(nums):
    nums.sort()
    res = []
    n = len(nums)
    used = [False]*n
    def backtrack(path):
        if len(path) == n:
            res.append(path[:])
            return
        for i in range(n):
            if used[i] or (i>0 and nums[i]==nums[i-1] and not used[i-1]):
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False
    backtrack([])
    return res
```

## 79. 单词搜索
- 思路：回溯+方向数组，标记访问
- 复杂度：**O(mn*3^L) / O(L)**
```python
def exist(board, word):
    m, n = len(board), len(board[0])
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    def dfs(i,j,k):
        if k == len(word): return True
        if i<0 or i>=m or j<0 or j>=n or board[i][j]!=word[k]:
            return False
        tmp = board[i][j]
        board[i][j] = '#'
        for dx, dy in dirs:
            if dfs(i+dx, j+dy, k+1): return True
        board[i][j] = tmp
        return False
    for i in range(m):
        for j in range(n):
            if dfs(i,j,0): return True
    return False
```

## 17. 电话号码的字母组合
- 思路：回溯，枚举数字对应字母
- 复杂度：**O(3^m *4^n) / O(m+n)**
```python
def letterCombinations(digits):
    if not digits: return []
    map = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
    res = []
    def backtrack(path, idx):
        if idx == len(digits):
            res.append(path)
            return
        for c in map[digits[idx]]:
            backtrack(path+c, idx+1)
    backtrack('', 0)
    return res
```

---

# 十、图 / BFS / 拓扑（8题）
## 200. 岛屿数量
- 思路：DFS/BFS遍历，淹没相连陆地
- 复杂度：**O(mn) / O(mn)**
```python
def numIslands(grid):
    if not grid: return 0
    m, n = len(grid), len(grid[0])
    cnt = 0
    def dfs(i, j):
        if i<0 or i>=m or j<0 or j>=n or grid[i][j]=='0':
            return
        grid[i][j] = '0'
        dfs(i+1,j),dfs(i-1,j),dfs(i,j+1),dfs(i,j-1)
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                cnt +=1
                dfs(i,j)
    return res
```

## 207. 课程表
- 思路：拓扑排序，BFS判断环
- 复杂度：**O(V+E) / O(V+E)**
```python
def canFinish(numCourses, prerequisites):
    from collections import deque, defaultdict
    in_degree = [0]*numCourses
    graph = defaultdict(list)
    for cur, pre in prerequisites:
        graph[pre].append(cur)
        in_degree[cur] +=1
    q = deque()
    for i in range(numCourses):
        if in_degree[i]==0: q.append(i)
    cnt =0
    while q:
        u = q.popleft()
        cnt +=1
        for v in graph[u]:
            in_degree[v] -=1
            if in_degree[v]==0: q.append(v)
    return cnt == numCourses
```

## 994. 腐烂的橘子
- 思路：多源BFS，层序遍历统计时间
- 复杂度：**O(mn) / O(mn)**
```python
def orangesRotting(grid):
    from collections import deque
    m, n = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                q.append((i,j))
            elif grid[i][j] == 1:
                fresh +=1
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    time = 0
    while q and fresh >0:
        size = len(q)
        for _ in range(size):
            x,y = q.popleft()
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                if 0<=nx<m and 0<=ny<n and grid[nx][ny]==1:
                    grid[nx][ny] = 2
                    fresh -=1
                    q.append((nx,ny))
        time +=1
    return time if fresh ==0 else -1
```

## 102. 二叉树的层序遍历
- 思路：队列BFS，按层收集
- 复杂度：**O(n) / O(n)**
```python
def levelOrder(root):
    if not root: return []
    from collections import deque
    q = deque([root])
    res = []
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level)
    return res
```

## 210. 课程表 II
- 思路：拓扑排序，记录遍历顺序
- 复杂度：**O(V+E) / O(V+E)**
```python
def findOrder(numCourses, prerequisites):
    from collections import deque, defaultdict
    in_degree = [0]*numCourses
    graph = defaultdict(list)
    for cur, pre in prerequisites:
        graph[pre].append(cur)
        in_degree[cur] +=1
    q = deque()
    for i in range(numCourses):
        if in_degree[i]==0:
            q.append(i)
    res = []
    while q:
        u = q.popleft()
        res.append(u)
        for v in graph[u]:
            in_degree[v] -=1
            if in_degree[v]==0:
                q.append(v)
    return res if len(res)==numCourses else []
```

## 127. 单词接龙
- 思路：双向BFS，最短路径
- 复杂度：**O(N*26^L) / O(N)**
```python
def ladderLength(beginWord, endWord, wordList):
    from collections import deque
    wordSet = set(wordList)
    if endWord not in wordSet: return 0
    q = deque([beginWord])
    wordSet.remove(beginWord)
    level = 1
    while q:
        size = len(q)
        for _ in range(size):
            word = q.popleft()
            if word == endWord: return level
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:i]+c+word[i+1:]
                    if new_word in wordSet:
                        wordSet.remove(new_word)
                        q.append(new_word)
        level +=1
    return 0
```

## 130. 被围绕的区域
- 思路：边界O标记，内部O替换为X
- 复杂度：**O(mn) / O(mn)**
```python
def solve(board):
    if not board: return
    m, n = len(board), len(board[0])
    def dfs(i,j):
        if i<0 or i>=m or j<0 or j>=n or board[i][j]!='O':
            return
        board[i][j] = '#'
        dfs(i+1,j),dfs(i-1,j),dfs(i,j+1),dfs(i,j-1)
    for i in range(m):
        dfs(i,0)
        dfs(i,n-1)
    for j in range(n):
        dfs(0,j)
        dfs(m-1,j)
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == '#':
                board[i][j] = 'O'
```

## 208. 实现 Trie (前缀树)
- 思路：多叉树节点，存储字符与结束标记
- 复杂度：**O(L) / O(26*L*N)**
```python
class Trie:
    def __init__(self):
        self.children = [None]*26
        self.is_end = False

    def insert(self, word):
        node = self
        for c in word:
            idx = ord(c)-ord('a')
            if not node.children[idx]:
                node.children[idx] = Trie()
            node = node.children[idx]
        node.is_end = True

    def search(self, word):
        node = self
        for c in word:
            idx = ord(c)-ord('a')
            if not node.children[idx]:
                return False
            node = node.children[idx]
        return node.is_end

    def startsWith(self, prefix):
        node = self
        for c in prefix:
            idx = ord(c)-ord('a')
            if not node.children[idx]:
                return False
            node = node.children[idx]
        return True
```
