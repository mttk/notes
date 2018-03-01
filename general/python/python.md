# Relevant python snippets

### Multiple inheritance and super()

-- Method resolution order (MRO) looks at the class's parents as they are listed **left to right**.

In the case of
```python
class Third(First, Second):
```

MRO will first check `First`, and if it doesn't have the attribute, it will look at `Second`.


```python
class First(object):
  def __init__(self):
    super(First, self).__init__()
    print "first"

class Second(object):
  def __init__(self):
    super(Second, self).__init__()
    print "second"

class Third(First, Second):
  def __init__(self):
    super(Third, self).__init__()
    print "that's it"
```

Method resolution order can be seen by `Object.__mro__`, but the order is Third -> First -> Second -> Object!

The super of init calls the leftmost arg in `Third`, which is `First`, but the super of `First` calls object, and then `Second` calls object. Duplicates are eliminated (object), leaving the rightmost, resulting in Third -> First -> Second -> object!

Useful: https://www.artima.com/weblogs/viewpost.jsp?thread=281127, https://stackoverflow.com/questions/3277367/how-does-pythons-super-work-with-multiple-inheritance

#### C3 linearization (python)
- Select the first **head** of the lists which does not appear in the **tail** in any of the lists
- Remove that element from all the lists where it is the head and append to output
- Repeat until the lists are empty


Use divide-and conquer to recursively find the linearizations of all the subclasses to use for the superclass.

https://en.wikipedia.org/wiki/C3_linearization

### Multiple inheritance, method overloads and super()

The case for method calls is different! The subclasses try to find the method from the leftmost class further on. Super calls in methods are searched according to the MRO!

### Mixin classes
Mixin classes are _init-less_ classes that add properties and methods into an existing class (with its own inheritance hierarchy)

https://stackoverflow.com/questions/533631/what-is-a-mixin-and-why-are-they-useful

