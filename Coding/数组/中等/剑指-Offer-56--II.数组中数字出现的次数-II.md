# [剑指 Offer 56 - II. 数组中数字出现的次数 II](https://leetcode.cn/problems/single-number-ii/description/)

---

## 题目

给你一个整数数组`nums`，除某个元素仅出现`一次`外，其余每个元素都恰出现`三次`。请你找出并返回那个只出现了一次的元素。  
你必须设计并实现线性时间复杂度的算法且使用常数级空间来解决此问题。  

示例 1：  
输入：nums = [2,2,3,2]  
输出：3  
示例 2：  
输入：nums = [0,1,0,1,0,1,99]  
输出：99  

提示：  

- 1 <= `nums.length` <= 3 * 104
- -231 <= `nums[i]` <= 231 - 1
- `nums` 中，除某个元素仅出现`一次`外，其余每个元素都恰出现`三次`

---

## 思路

统计所有数每个`bit位 出现的 次数`。  
再把每个位的`次数 % 3`，也就算出，只出现1次数的`bit位`。  
最后再`各个 bit 位拼接`起来，就得到了只出现 1次 的数  

---

## 代码

```C++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;
        for (int i = 0, sub = 0; i < 32; ++i, sub = 0) {
            for (auto &n : nums)
                sub += ((n >> i) & 1);
            if (sub % 3)
                res |= (1 << i);
        }
        return res;
    }
};
```

时间复杂度：**O(n)**  
空间复杂度：**O(1)**
