# [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode.cn/problemset/all/)

---

## 题目

一个整型数组`nums`里除`两个数字只出现了一次`之外，`其他数字都出现了两次`。请写程序`找出这两个只出现一次的数字`。

---

## 思路

位运算：异或  
位异或，相异为1，相同为0.  
注意：`a ^ a ^ b ^ b ^ c = c`

---

## 代码

```C++
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        int ret = 0;
        for(int i:nums)
            ret ^=i; // 最终ret是两个只出现一次的数的异或结果
        int div = 1;
        while((ret&div)==0) // 找到ret最右边为1的那一位，两个只出现一次的数在这一位上是不同的，记为第x位
            div = (div << 1); 
        int a=0,b=0;
        for(int i:nums){ // 第x位为0的异或到a上，为1的异或到b上
            if((i&div) == 0)
                a ^= i;
            else
                b ^= i;
        }
        return vector<int>{a,b}; // 最终a和b就是只出现一次的两个数字
    }
};
```

时间复杂度：**O(n)**  
空间复杂度：**O(1)**
