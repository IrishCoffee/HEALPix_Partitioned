#ifndef STL_CUDA_H
#define STL_CUDA_H

class Pair
{
  public:
	__device__ Pair()
	{
	  first = 0;
	  second = 0;
	}
	__device__ Pair(int first_,int second_)
	{
	  first = first_;
	  second = second_;
	}
	int first;
	int second;
};

class Pair_Vector
{
  public:
	__device__ Pair_Vector()
	{
	  top = -1;
	  size = 0;
	}

	__device__ void push_back(Pair &item)
	{
	  top++;
	  pair_vector[top].first = item.first;
	  pair_vector[top].second = item.second;
	  size++;
	  return;
	}

	__device__ void append(int v)
	{
	  append(v,v+1);
	  return;
	}

	__device__ void append(int v1, int v2)
	{
	  if(v2 <= v1)
		return;
	  if((!isEmpty()) && (v1 <= back().second))
	  {
		if(v2 > back().second)
		  pair_vector[top].second = v2;
	  }
	  else
	  {
		top++;
		pair_vector[top].first = v1;
		pair_vector[top].second = v2;
		size++;
	  }
	}

	__device__ void add(int v,bool &flag)
	{
	  if(flag)
	  {
		top++;
		pair_vector[top].first = v;
		size++;
		flag = false;
	  }
	  else
	  {
		pair_vector[top].second = v;
		flag = true;
	  }
	}

	__device__ void clear()
	{
	  size = 0;
	  top = -1;
	}

	__device__ void toVector(Pair_Vector &r)
	{
	  r.clear();
	  bool flag = true;
	  for(int i = 0; i < getSize(); i++)
	  {
		for(int m = pair_vector[i].first; m < pair_vector[i].second; ++m)
		  r.add(m,flag);
	  }
	}

	__device__ Pair back()
	{
	  Pair temp;
	  temp.first = pair_vector[top].first;
	  temp.second = pair_vector[top].second;
	  return temp;
	}

	__device__ void pop_back()
	{
	  top--;
	  size--;
	  return;
	}

	__device__ bool isEmpty()
	{
	  if(size == 0)
		return true;
	  return false;
	}

	__device__ int getSize()
	{
	  return size;
	}

	__device__ void resize(int size_)
	{
	  size = size_;
	  top = size - 1;
	  return;
	}
	Pair pair_vector[32];
	int top;
	int size;
};

class Stack
{
  public:
	__device__ Stack()
	{
	  top = -1;
	  size = 0;
	}

	__device__ void push_back(Pair &item)
	{
	  top++;
	  pair_vector[top].first = item.first;
	  pair_vector[top].second = item.second;
	  size++;
	  return;
	}

	__device__ void clear()
	{
	  size = 0;
	  top = -1;
	}

	__device__ Pair back()
	{
	  Pair temp;
	  temp.first = pair_vector[top].first;
	  temp.second = pair_vector[top].second;
	  return temp;
	}

	__device__ void pop_back()
	{
	  top--;
	  size--;
	  return;
	}

	__device__ bool isEmpty()
	{
	  if(size == 0)
		return true;
	  return false;
	}

	__device__ int getSize()
	{
	  return size;
	}

	__device__ void resize(int size_)
	{
	  size = size_;
	  top = size - 1;
	  return;
	}
	Pair pair_vector[64]; // size = 12 + omax * 3
	int top;
	int size;
};

#endif
