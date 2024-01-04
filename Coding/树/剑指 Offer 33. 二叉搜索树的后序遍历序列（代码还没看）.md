# [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/description/)

---

## 题目 (中等)

请实现一个函数来判断整数数组 postorder 是否为二叉搜索树的后序遍历结果。  

示例 1：  
![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_33_1.png)  

```markdown
输入: postorder = [4,9,6,9,8]
输出: false
解释：从上图可以看出这不是一颗二叉搜索树
```

示例 2：  
![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_33_2.png)  

```markdown
输入: postorder = [4,6,5,9,8]
输出: true
解释：可构建的二叉搜索树如上图
```

提示：  

- 数组长度 <= 1000
- postorder 中无重复数字

---

## 思路

---

## 代码

```C++
#include <template.h>
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    bool check(vector<int>& postorder,int L,int R){// 检查区间[L,R]是否是后序遍历序列
        if(L>=R)
            return true;
        int n = L;
        while(postorder[n]<postorder[R])n++;
        int m = n;
        while(postorder[n]>postorder[R])n++;
        return R==n && check(postorder,L,m-1) && check(postorder,m,R-1);// 分别检查左右子树区间是否是后序遍历序列
    }
    bool verifyPostorder(vector<int>& postorder) {  
        return check(postorder,0,postorder.size()-1);
    }
};
```

时间复杂度：**O(n)**  
空间复杂度：**O(1)**
