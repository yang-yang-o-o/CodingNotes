# [剑指Offer 31.栈的压入弹出序列](https://leetcode.cn/problems/validate-stack-sequences/description/)

---

## 题目 (中等)

给定 pushed 和 popped 两个序列，每个序列中的 值都不重复，只有当它们可能是在最初空栈上进行的推入 push 和弹出 pop 操作序列的结果时，返回 true；否则，返回 false 。  

示例 1：  

```markdown
输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
```

示例 2：  

```markdown
输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。
```

提示：  

- 1 <= pushed.length <= 1000
- 0 <= pushed[i] <= 1000
- pushed 的所有元素 互不相同
- popped.length == pushed.length
- popped 是 pushed 的一个排列

---

## 思路

---

## 代码

```C++
class Solution31 {
    /*
    用一个新栈s来实时模拟进出栈操作：

    在for里依次喂数，每push一个数字就检查有没有能pop出来的。

    如果最后s为空，说明一进一出刚刚好。

    时间复杂度分析：一共push n次，pop n次。
    */
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        if(pushed.size() != popped.size()) return false;
        stack<int> stk;
        int i = 0;
        for(auto x : pushed){
            stk.push(x);
            while(!stk.empty() && stk.top() == popped[i]){
                stk.pop();
                i++;
            }
        }
        return stk.empty();
    }
};
// 自己写的解法
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        if(pushed.size()!=popped.size())
            return false;
        stack<int> tmp;
        int n = popped.size();
        int j=0;
        for(int i=0;i<pushed.size();i++){
            tmp.push(pushed[i]);
            while(!tmp.empty() && j<n && tmp.top() == popped[j]){// 这里!tmp.empty() 和 j<n 其实是等价的，j<n可以不要，
                                                                 // 但是!tmp.empty()一定要，这是因为这个条件一定是先不成立的
                tmp.pop();
                j++;
            }
        }
        return j==n;
    }
};
```
