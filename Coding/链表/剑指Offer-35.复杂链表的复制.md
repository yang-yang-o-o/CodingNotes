# [剑指 Offer 35. 复杂链表的复制](https://leetcode.cn/problems/fu-za-lian-biao-de-fu-zhi-lcof/description/)

---

## 题目 (中等)

请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。  

示例 1：  
![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_35_1.png)  

```markdown
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
```

示例 2：  
![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_35_2.png)  

```markdown
输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]
```

示例 3：  
![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_35_3.png)  

```markdown
输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]
```

示例 4：  

```markdown
输入：head = []
输出：[]
解释：给定的链表为空（空指针），因此返回 null。
```

提示：  

- -10000 <= Node.val <= 10000
- Node.random 为空（null）或指向链表中的节点。
- 节点数目不超过 1000 。

注意：本题与[主站 138 题](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)相同  

---

## 思路

递归解法：
    复制当前节点，然后分别调用递归复制next节点和random节点，  
    递归终止条件有两个：如果要复制的节点为nullptr，就返回nullptr，如果不为nullptr且已经复制过了，返回mp[head]  
    ```unordered_map<Node*,Node*> mp; // 存储已经完成复制的节点对```

迭代解法：
    在原始链表的每个节点后都复制和插入前一个节点，总的节点个数翻倍，  
    然后复制random节点：```ptr->next->random = (ptr->random==nullptr)?nullptr:ptr->random->next;```  
    然后将链表按奇数和偶数拆成两个，即完成链表的复制。

---

## 代码

### 1. 递归解法

```C++
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
class Solution {               // 递归解法
public:
    unordered_map<Node*,Node*> mp; // 存储已经完成复制的节点对
    Node* copyRandomList(Node* head) {
        //// 递归终止条件
        if(head==nullptr)
            return head;
        if(mp.find(head)!=mp.end()) // 如果已经复制过了
            return mp[head];
        //// 复制节点
        Node* node = new Node(head->val);
        mp[head] = node;
        node->next = copyRandomList(head->next);
        node->random = copyRandomList(head->random);
        return node;
    }
};
```

### 2. 迭代解法

![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_35_4.png)

```C++
class Solution {            // 迭代解法
public:
    Node* copyRandomList(Node* head) {
        if(head==nullptr)
            return head;
        Node* ptr = head;
        while(ptr!=nullptr){// 复制节点
            Node* tmp = new Node(ptr->val);
            tmp->next = ptr->next;
            ptr->next = tmp;
            ptr = ptr->next->next;
        }
        ptr = head;
        while(ptr!=nullptr){// 复制random
            ptr->next->random = (ptr->random==nullptr)?nullptr:ptr->random->next;
            ptr = ptr->next->next;
        }
        Node* oldlist = head;
        Node* newlist = head->next;
        Node* tmp = head->next;
        while(oldlist!=nullptr){// 拆分
            oldlist->next = oldlist->next->next;
            newlist->next = (newlist->next==nullptr)?nullptr:newlist->next->next;
            oldlist = oldlist->next;
            newlist = newlist->next;
        }
        return tmp;
    }
};
```
