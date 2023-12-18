# [剑指 Offer 03. 数组中重复的数字](https://leetcode.cn/problemset/all/)

---

## 题目



---

## 思路

原地置换

---

## 代码

```C++
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        for(int i=0;i<nums.size();i++)
            while(nums[i]!=i){// 遍历每一个元素，只要下标和元素不等就去不停的交换
                if(nums[i] == nums[nums[i]])// 如果当前的元素已经和要换的元素相等了，就找到重复了
                    return nums[i];
                swap(nums[i],nums[nums[i]]);
            }
        return -1;
    }
};
```

时间复杂度：**O(n)**  
空间复杂度：**O(1)**
