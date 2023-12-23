# [剑指 Offer 44. 数字序列中某一位的数字](https://leetcode.cn/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/description/)

---

## 题目 (中等)

某班级学号记录系统发生错乱，原整数学号序列 [0,1,2,3,4,...] 分隔符丢失后变为 01234... 的字符序列。请实现一个函数返回该字符序列中的第 k 位数字。  

示例 1：  

```markdown
输入：k = 5
输出：5
```

示例 2：  

```markdown
输入：k = 12
输出：1
解释：第 12 位数字在序列 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... 里是 1 ，它是 11 的一部分。
```

提示：  

- 0 <= k < 231

注意：本题与[主站 400 题](https://leetcode-cn.com/problems/nth-digit/)相同

---

## 思路

---

## 代码

```C++
class Solution {
public:
    int findNthDigit(int n) {
        int digit = 1;
        long start = 1;
        long count = 9;
        while(n>count){// 第一次循环查看一位的所有数，第二次循环看两位的所有数，第三次循环看三位的所有数。。。
            n -= count;
            digit += 1;
            start *= 10;
            count = 9*digit*start;// 两位数从10到99，总的90个数，每个数是两个字符，就是180
        }
        long num = start + (n-1)/digit;// 第n位是下标n-1，(n-1)/digit计算前n位包含多少个完整的数，加上start得到的是以前n位除以digit余下的位数开始的完整的数
        string nums = to_string(num);// 第n位所在的数字转换为字符串
        int res = nums[(n-1)%digit] - '0';// 在字符串中找出第n位，因为第n位是下标n-1
        return res;
    }
};
```
