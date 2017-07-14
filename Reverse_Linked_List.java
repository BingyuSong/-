public class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; }
}
public class Solution {
    public ListNode reverseList(ListNode head) {
       if(head==null || head.next==null)
        return head;
       ListNode nextnode=head.next;
       ListNode newnode=reverseList(nextnode);
       nextnode.next=head;
       head.next=null;
       return newnode;
        

    }
}
