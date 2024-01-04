# [剑指 Offer 26. 树的子结构](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/description/)

---

## 题目 (中等)

给定两棵二叉树 tree1 和 tree2，判断 tree2 是否以 tree1 的某个节点为根的子树具有 相同的结构和节点值 。  
注意，空树 不会是以 tree1 的某个节点为根的子树具有 相同的结构和节点值 。  

示例 1：  
![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_26_1.png)  

```markdown
输入：tree1 = [1,7,5], tree2 = [6,1]
输出：false
解释：tree2 与 tree1 的一个子树没有相同的结构和节点值。
```

示例 2：  
![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_26_2.png)  

```markdown
输入：tree1 = [3,6,7,1,8], tree2 = [6,1]
输出：true
解释：tree2 与 tree1 的一个子树拥有相同的结构和节点值。即 6 - > 1。
```

提示：  

- 0 <= 节点个数 <= 10000

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
    bool check(TreeNode* p,TreeNode* q){
        if(p && q && p->val==q->val)
            return check(p->left,q->left) && check(p->right,q->right);
        else if(!q)// 这个条件和572. 另一个树的子树不同，其它的代码是相同的，这里的含义是，只要q为空，不管p为不为空，都返回true。
            return true;
        else 
            return false;
    }
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if(A && B)
            return check(A,B) || isSubStructure(A->left,B) || isSubStructure(A->right,B);
        else if(!A && !B)
            return true;
        else 
            return false;
    }
};
```
