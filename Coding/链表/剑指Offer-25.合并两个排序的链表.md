# [剑指 Offer 25. 合并两个排序的链表](https://leetcode.cn/problems/merge-two-sorted-lists/description/)

---

## 题目 (简单)

将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。  

示例 1：  
![Alt text](https://github.com/yang-yang-o-o/CodingNotes/blob/main/Coding/asset/offer_25_1.png)  

```markdown
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

示例 2：  

```markdown
输入：l1 = [], l2 = []
输出：[]
```

示例 3：  

```markdown
输入：l1 = [], l2 = [0]
输出：[0]
```

提示：  

- 两个链表的节点数目范围是 [0, 50]
- -100 <= Node.val <= 100
- l1 和 l2 均按 非递减顺序 排列

---

## 思路

双指针：两个指针分别指向两个链表，然后不断的比较值大小，小的拿出来排序，然后小的指针往后移

---

## 代码

```C++
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) { // 合并两个有序链表
        ListNode* head = new ListNode(); // 虚拟头结点
        ListNode* tmp = head;
        while(l1!=nullptr && l2!=nullptr){ // 使用双指针的形式合并
            if(l1->val < l2->val){
                tmp->next = l1;
                l1 = l1->next;
            }
            else{
                tmp ->next = l2;
                l2 = l2->next;
            }
            tmp = tmp->next;
        }
        // 将某个链表剩下的节点直接接到结果尾部
        if(l1!=nullptr)
            tmp->next = l1;
        if(l2!=nullptr)
            tmp->next = l2;
        return head->next; // 返回合并后的有序链表
    }
};
```

time：O(n+m)
space：O(1)
