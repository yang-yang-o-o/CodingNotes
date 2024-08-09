# [剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode.cn/problems/sliding-window-maximum/)

---

## 题目 (hard)

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。  

返回 `滑动窗口中的最大值`。  

示例 1：  

```markdown
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

示例 2：  

```markdown
输入：nums = [1], k = 1
输出：[1]
```

提示：  

- 1 <= `nums.length` <= 10e5
- -10e4 <= `nums[i]` <= 10e4
- 1 <= `k` <= nums.length

---

## 思路

利用双端队列构造一个递减单调队列，队列首始终放当前窗口中的最大值，  

滑动窗口每滑动一步，

- 如果窗口左边界划过的要丢弃的元素等于队列首元素，队列首元素就出队，
- 如果窗口右边界划过的要新增的元素比队列尾元素大，队列尾元素出队，直到新增元素`小于等于`队列尾元素`或者队列为空`，再将新元素从尾部入队。

---

## 代码

```C++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;

        if (nums.size() == 0) return res;

        deque<int> Q;
        for (int i = 0; i < k; i++){ // 构建从左往右的第一个窗口
            while (!Q.empty() && Q.back() < nums[i]) // 注意这里不能是小于等于
                Q.pop_back();
            Q.push_back(nums[i]);
        }
        res.push_back(Q.front());

        for (int i = k; i < nums.size(); i++) { // 开始滑动窗口
            // 考虑左边界划过的要丢弃的元素
            if (nums[i-k] == Q.front())
                Q.pop_front();
            // 考虑右边界划过的要新增的元素
            while (!Q.empty() && Q.back() < nums[i])  // 注意这里不能是小于等于
                Q.pop_back();                      // vector<int> nums = {-7,-8,7,5,7,1,6,0};  int k = 4;
                                                   // 如果是小于等于，第二个7加入q时会把q中第一个7删掉，而窗口划过第一个7时，本该删除q中第一个7，但由于等号存在，第一个7已被删除，此时只能删除第二个7，后续的结果就出错
            Q.push_back(nums[i]);
            res.push_back(Q.front());
        }
        return res;
    }
};
```

时间复杂度：**O( n )**  
空间复杂度：**O( n )**
