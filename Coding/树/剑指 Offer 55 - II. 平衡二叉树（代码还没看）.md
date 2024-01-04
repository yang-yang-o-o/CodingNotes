# [剑指 Offer 55 - II. 平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/description/)

---

## 题目 (简单)

给定一个二叉树，判断它是否是高度平衡的二叉树。  

本题中，一棵高度平衡二叉树定义为：  

一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。  

示例 1：  
![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_55_1.png)  

```markdown
输入：root = [3,9,20,null,null,15,7]
输出：true
```

示例 2：  
![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_55_2.png)

```markdown
输入：root = [1,2,2,3,3,null,null,4,4]
输出：false
```

示例 3：  

```markdown
输入：root = []
输出：true
```

提示：  

- 树中的节点数在范围 [0, 5000] 内
- -10^4 <= Node.val <= 10^4

---

## 思路

深度优先，递归，同110题

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
    int check(TreeNode*root)
    {
        if(!root)return 0;
        int l,r;
        if((l=check(root->left))==-1 || (r=check(root->right))==-1 || abs(l-r)>1)return -1;
        else return max(l,r)+1;
    }
    bool isBalanced(TreeNode* root) {
        return check(root)!=-1;
    }
};
// 自己写的解法
class Solution {
public:
    int balance(TreeNode*root){
        if(root==nullptr)return 0;
        int l = balance(root->left);
        int r = balance(root->right);
        if(l == -1 || r == -1 || abs(l-r)>1)// 这里可以像上面那种解法一样优化
            return -1;
        return max(l,r)+1;
    }
    bool isBalanced(TreeNode* root) {
        return balance(root)!=-1;        
    }
};
```

时间复杂度：**O(n)**  
空间复杂度：**O(n)**
