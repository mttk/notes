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

Useful: https://en.wikipedia.org/wiki/C3_linearization, https://www.artima.com/weblogs/viewpost.jsp?thread=281127


### Multiple inheritance, method overloads and super()

The case for method calls is different! The subclasses try to find the method from the leftmost class further on. Super calls in methods are searched according to the MRO!
