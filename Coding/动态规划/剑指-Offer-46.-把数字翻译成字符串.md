# [剑指 Offer 46. 把数字翻译成字符串](https://leetcode.cn/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

---

## 题目 (中等)

现有一串神秘的密文 `ciphertext`，经调查，密文的特点和规则如下：  

- 密文由非负整数组成
- 数字 0-25 分别对应字母 a-z
请根据上述规则将密文 `ciphertext` 解密为字母，并返回共有多少种解密结果。  

示例 1:  

输入: ciphertext = 216612  
输出: 6  
解释: 216612 解密后有 6 种不同的形式，分别是 "cbggbc"，"vggbc"，"vggm"，"cbggm"，"cqggbc" 和 "cqggm"  

提示：  

- 0 <= ciphertext < 2^31

---

## 思路

动态规划

---

## 代码

```C++
class Solution {
public:
    int translateNum(int num) {
        string s = to_string(num);
        int pre = 1,prepre = 0,cur = 0;// pre表示以上一个字符结尾的字符串可以翻译的数量
                                       // prepre表示以上一个的上一个字符结尾的字符串可以翻译的数量
                                       // cur表示以当前字符结尾的字符串可以翻译的数量   
        for(int i=0;i<s.size();i++){
            cur = pre;
            if(i>0 && s.substr(i-1,2)>="10" && s.substr(i-1,2)<="25")
                cur += prepre;
            prepre = pre;
            pre = cur;
        }
        return cur;
    }
};
```

时间复杂度：**O(n)**  
空间复杂度：**O(1)**
