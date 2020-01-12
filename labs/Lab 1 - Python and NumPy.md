---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
Alex Tschopp


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


**YOUR ANSWER HERE**


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1

```python
import numpy as np
a = np.ones((6, 4), int) * 2
```

## Exercise 2

```python
b = np.ones((6, 4), int) 
np.fill_diagonal(b, 3)
```

## Exercise 3


The '*' operator multiplies the arrays 'a' and 'b' element-wise while the function np.dot(a,b) requires the number of columns of 'a' to be equal to the number of rows of 'b'. The element-wise multiplication works because the matrices have the same shape and the dot product does not work because the shapes of the matrices are incompatible.


## Exercise 4


The results of these dot products are different because the outer dimensions of the two matrices used to compute the dot products are different. Computing a dot product with shapes (4 x 6) dot (6 x 4) yields a (4 x 4) matrix while performing (6 x 4) dot (4 x 6) yields a (6 x 6) matrix.


## Exercise 5

```python
def print_arr(arr):
    print(arr)
print_arr(a)
```

## Exercise 6

```python
def random_computations(*shape):
    arr = np.random.rand(*shape)
    print('sum:', np.sum(arr))
    print('mean:', np.mean(arr))
random_computations(3,3)
```

## Exercise 7

```python
def count_ones_with_loops(arr):
    numOfOnes = 0
    shape = arr.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (arr[i][j] == 1):
                numOfOnes += 1
    return numOfOnes

def count_ones_with_where(arr):
    isOne = np.where(arr == 1)
    return len(arr[isOne])
```

## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
import pandas as pd
adf = pd.DataFrame(np.ones((6, 4), int) * 2)
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
bdf = pd.DataFrame(np.ones((6,4), int))
bdf.iloc[0,0] = 2
bdf.iloc[1,1] = 2
bdf.iloc[2,2] = 2
bdf.iloc[3,3] = 2
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
# Element-wise multiplication works because they are the same shape, as before
print(adf * bdf)
try:
    # Inner dimensions do not match, as was the case in A.3
    print(adf.dot(bdf))
except Exception as e:
    print(e)
```

## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
def count_ones_with_loops(df):
    numOfOnes = 0
    shape = df.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (df.iloc[i][j] == 1):
                numOfOnes += 1
    return numOfOnes

def count_ones_with_where(df):
    isOne = df.where(df == 1)
    return isOne.count().sum()
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
titanic_df['name']
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
titanic_df.set_index('sex',inplace=True)
titanic_df.loc['female']
```

## Exercise 14
How do you reset the index?

```python
titanic_df.reset_index()
```
