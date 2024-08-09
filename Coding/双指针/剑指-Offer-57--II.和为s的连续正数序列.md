# [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode.cn/problemset/all/)

---

## 题目 (简单)

输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。  

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。  

示例 1：  
```
输入：target = 12
输出：[[3, 4, 5]]
解释：在上述示例中，存在一个连续正整数序列的和为 12，为 [3, 4, 5]。
```

示例 2：  
```
输入：target = 18
输出：[[3,4,5,6],[5,6,7]]
解释：在上述示例中，存在两个连续正整数序列的和分别为 18，分别为 [3, 4, 5, 6] 和 [5, 6, 7]。
```

---

## 思路

---

## 代码

```C++
class Solution {
public:
    vector<vector<int>> findContinuousSequence(int target) {
        vector<vector<int>> res;
        int l=1, r=2; // 正整数从1,2开始
        while(l<r){// 类似于滑动窗口，实际上这个滑动窗口枚举的是左边界
            int sum = (l+r)*(r-l+1)/2;// 等差数列求和
            if(sum == target){
                vector<int> tmp;
                for(int i=l;i<=r;i++)
                    tmp.push_back(i);
                res.push_back(tmp);
                l++;// 移动窗口左边界，此时移动右边界不可能再等于target，只会更大
            }
            else if(sum > target)// 窗口内和大于目标值，移动左边界，小于移动右边界
                l++;
            else 
                r++;
        }
        return res;
    }
};
```

时间复杂度：**O(n)**  
空间复杂度：**O(1)**
