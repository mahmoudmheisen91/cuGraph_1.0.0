#ifndef STACKPOINTER_H_
#define STACKPOINTER_H_


class StackPointer
{
	public:
		StackPointer();
		int getSize();
	protected:
	private:
		int size;
		int item;
		int *next;
};

#endif // STACKPOINTER_H_
