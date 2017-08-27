
#include "Stack.h"
using namespace std;

//   Creating a NODE Structure

// Creating a class STACK

// PUSH Operation
CUDA_HOSTDEV void Stack::push(const CUDA_KDNode a, float d)
{
	
	struct node *ptr;
	ptr = new node;
	ptr->dsq = d;
	ptr->kdnode = a;
	if (top != NULL)
		ptr->next = top;
	top = ptr;
	size++;
}
CUDA_HOSTDEV Stack::Stack(){
	size = 0;
}
// POP Operation
CUDA_HOSTDEV CUDA_KDNode Stack::pop()
{
	struct node *temp;
	if (top == NULL)
	{
		cout << "nThe stack is empty!!!";
	}
	temp = top;
	top = top->next;
	size--;
	return temp->kdnode;
}
