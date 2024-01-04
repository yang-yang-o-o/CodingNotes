# [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/)

---

## 题目 (中等)

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。  

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”  

示例 1：  
![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_68_1.png)  

```markdown
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。
```

示例 2：  

![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_68_2.png)  

```markdown
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。
```

示例 3：  

```markdown
输入：root = [1,2], p = 1, q = 2
输出：1
```

提示：

- 树中节点数目在范围 [2, 10^5] 内。
- -10^9 <= Node.val <= 10^9
- 所有 Node.val 互不相同 。
- p != q
- p 和 q 均存在于给定的二叉树中。

---

## 思路

递归后序遍历————经典题目
        解题（不止于此题）的关键在于枚举出所有可能的情况，只有考虑了所有的情况，才能设计出正确的算法。————切记
    比如此题，只有两种种情况：
        1、p和q都不为对方的祖先，此时p和q必然在某个节点的两侧，那么返回这个节点即可
        2、p和q中其中一个是另一个的祖先，那么返回这个祖先即可。
    基于以上两种情况，怎么在递归时实现：
        对于第一种情况，回到当前递归层时怎么知道q在root的一个子树上，p在另外一个子树上，所以肯定需要递归返回一个标记，
        来表示这棵子树上找到了一个节点，因为p和q在两侧的两种情况都满足情况1，因此也就不需要知道一个子树的到底是找到了p还是q，
        所以这个标记可以是任意的标记，子树最好是设为返回找到的那个节点，因为这样可以简化编程，程序最终需要的是祖先。
        对于第二种情况，如果当前节点是p或者q，就不需要再去看他的子树，递归返回过程中，如果其他的子树上没有要找的另一个节点，
        那么另一个节点必然在刚刚那个当前节点的子树上，那么那个当前节点就是祖先，程序需要的是他，这也是为什么前面提到为什么返回的标记
        最好设为找到的节点，这样对于这种情况，直接返回，不需要其他的转换工作。

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
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
       if(!root || root == p || root == q)return root;//巧妙的地方，如果当前的节点是pq之一，那么子树就不用找了，直接返回，去其他的子树找，其他子树没有，那另一个必然在这个子树中，程序需要的也就是当前的root
       TreeNode* l = lowestCommonAncestor(root->left,p,q);// 去左子树找p或者q中的一个，找到就返回结点，没找到就返回nullptr
       TreeNode* r = lowestCommonAncestor(root->right,p,q);// 去右子树找p或者q中的一个，找到就返回结点，没找到就返回nullptr
       // 下面的三条如果l或者r为空，返回另一个，为空说明这个子树没有p或者q
       // 如果两个都为空，返回谁都可以
       // 如果都不为空，说明是情况1,返回root即可
       if(!l)return r;
       if(!r)return l;// 这两句秒在既处理了情况1中在一个子树找p或者q的返回过程，也处理了情况2中通过 去另外的子树找p或者q中的另一个来推断另一个是否在刚刚找到的其中一个节点的子树中。
       return root;
    }
};
// 自己写的解法
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == nullptr || root == p || root == q)return root;
        TreeNode* l = lowestCommonAncestor(root->left,p,q);
        TreeNode* r = lowestCommonAncestor(root->right,p,q);
        if(l==nullptr)
            return r;
        if(r==nullptr)
            return l;
        return root;
    }
};

```

时间复杂度：**O(n)**  
空间复杂度：**O(n)**
