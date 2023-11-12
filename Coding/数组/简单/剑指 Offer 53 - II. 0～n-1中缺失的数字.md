# [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode.cn/problems/que-shi-de-shu-zi-lcof/description/)

---

## 题目

数组`nums`长度为`n`，按升序存放了`[0,n]`内的`n`个数，找出`[0,n-1]`中没有出现的数

示例 1:  
输入: records = [0,1,2,3,5]  
输出: 4  

示例 2:  
输入: records = [0,1,2,3,4,5,6,8]  
输出: 7  

---

## 思路

二分查找

---

## 代码

```C++
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        if(nums.empty()) return 0;
        int l = 0, r = nums.size() - 1;
        while(l <= r){
            int mid = (l + r) >> 1;
            if(nums[mid] != mid) 
                r = mid-1; // 只要下标i不等于值nums[i]就到左边去继续找，最终找到的就是第一个和值不相等的下标。
            else 
                l = mid + 1;
        }
        return l;
    }
};
```

时间复杂度：**O(logn)**  
空间复杂度：**O(1)**
