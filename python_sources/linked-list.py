# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# Linked lists are like a lesser-known cousin of lists. 
# They’re not as popular or as cool, and you might not even remember them from your algorithms class.

# Linked lists are an ordered collection of objects. 
# So what makes them different from normal lists? Linked lists differ from lists in the way that they store elements in memory. 
# While lists use a contiguous memory block to store references to their data, linked lists store references as part of their own elements.

class Box:
  def __init__ (self,cat = None):
    self.cat = cat
    self.nextcat = None

class LinkedList:
  def __init__(self):
    self.head = None

  def contains (self, cat):
    lastbox = self.head
    while (lastbox):
      if cat == lastbox.cat:
        return True
      else:
        lastbox = lastbox.nextcat
    return False

  def addToEnd(self, newcat):
    newbox = Box(newcat)
    if self.head is None:
      self.head = newbox
      return
    lastbox = self.head
    while (lastbox.nextcat):
        lastbox = lastbox.nextcat
    lastbox.nextcat = newbox


  def get(self, catIndex):
    lastbox = self.head
    boxIndex = 0
    while boxIndex <= catIndex:
      if boxIndex == catIndex:
          return lastbox.cat
      boxIndex = boxIndex + 1
      lastbox = lastbox.nextcat

  def removeBox(self,rmcat):
    headcat = self.head

    if headcat is not None:
      if headcat.cat==rmcat:
        self.head = headcat.nextcat
        headcat = None
        return
    while headcat is not None:
      if headcat.cat==rmcat:
        break
      lastcat = headcat
      headcat = headcat.nextcat
    if headcat == None:
      return
    lastcat.nextcat = headcat.nextcat
    headcat = None


  def LLprint(self):
    currentCat = self.head
    print("LINKED LIST")
    print("-----")
    i = 0
    while currentCat is not None:
      print (str(i) + ": " + str(currentCat.cat))
      i += 1
      currentCat = currentCat.nextcat
    print("-----")



