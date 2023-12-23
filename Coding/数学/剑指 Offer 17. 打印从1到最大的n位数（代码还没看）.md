# [剑指 Offer 17. 打印从1到最大的n位数](https://leetcode.cn/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/description/)

---

## 题目 (简单)

实现一个十进制数字报数程序，请按照数字从小到大的顺序返回一个整数数列，该数列从数字 1 开始，到最大的正整数 cnt 位数字结束。  

示例 1:  

```markdown
输入：cnt = 2
输出：[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]
```

---

## 思路

---

## 代码

```C++
// 自己写的解法，使用全排列
class Solution {
public:
    void dfs(vector<string>& res,string& tmp,int idx){// 注意需要按引用传递
        if(idx == tmp.size()){
            int start = 0;
            for(int i=0;i<tmp.size();i++)// 去除高位无效0
                if(tmp[i]=='0')
                    start++;
                else
                    break;
            res.push_back(tmp.substr(start));
            return;
        }
        for(int i=0;i<=9;i++){// 每一位有9种选择
            tmp[idx] = (char)(i+'0');// 注意这种字符构造方法
            dfs(res,tmp,idx+1);
        }
    }
    vector<string> printNumbers(int n) {
        vector<string> res;
        string tmp(n,'0');// 注意这种构造方法
        dfs(res,tmp,0);
        return res;
    }
};
```
