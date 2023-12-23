# [剑指 Offer 61. 扑克牌中的顺子](https://leetcode.cn/problems/bu-ke-pai-zhong-de-shun-zi-lcof/description/)

---

## 题目 (简单)

展览馆展出来自 13 个朝代的文物，每排展柜展出 5 个文物。某排文物的摆放情况记录于数组 places，其中 places[i] 表示处于第 i 位文物的所属朝代编号。其中，编号为 0 的朝代表示未知朝代。请判断并返回这排文物的所属朝代编号是否连续（如遇未知朝代可算作连续情况）。  

示例 1：  
```
输入: places = [0, 6, 9, 0, 7]
输出: True
```

示例 2：  
```
输入: places = [7, 8, 9, 10, 11]
输出: True
```

提示：  

- places.length = 5
- 0 <= places[i] <= 13

---

## 思路

---

## 代码

```C++
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        int joker = 0;
        sort(nums.begin(),nums.end());// 数组排序
        for(int i=0;i<4;++i){
            if(nums[i] == 0) joker++; // 统计大小王数量
            else if(nums[i] == nums[i+1])return false;// 若有重复，提前返回 false
        }
        return nums[4] - nums[joker] <5;// 最大牌 - 最小牌 < 5 则可构成顺子
    }
};
```

时间复杂度：**O(n)**  
空间复杂度：**O(1)**
