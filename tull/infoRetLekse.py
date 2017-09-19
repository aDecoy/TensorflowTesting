#2.7 The answers on the blackboard did not make sence, so i investigated a bit. The blackboard answers was 1 skip (correct), b) 10 and c)18
# turns out there are 33 comparisons without skipointers and 32 with skip pointers
#Note that "and"-operations save some time. example in if(False and code_Will_Never_Run()), the function will never run since the expression has to be False anyway

#by posting comparisjon i included both when they are compared to be the same,     [16 times with skip pointers]
# when they are compared to find who are smalest, and also when the next skip pointer value is compared with the value of list2   [16 times with skip pointers]



list1=(3,5,9,15,24,39,60,68,75,81,84,89,92,96,97,100,115)
list2=(3,5,89,95,97,99,100,101)

skipPointers=[3,24,75,92,115]
# skipPointers=[]

equalComparisons= 0
sizeComparison= 0

a=4
b=5

def test(value,match):
    print('test function ran')
    if value==match:
        return True
    return False

if (False and test(a,4)):
    print('teset fullført')

def lower(value1,value2):
    global sizeComparison
    sizeComparison+=1
    # print('sizecomparison', siceComparison)
    return value1<value2
def lowerOrEqual(value1,value2):
    global sizeComparison
    sizeComparison+=1
    # print('sizecomparison', siceComparison)

    return value1<=value2

def hasSkip(value):
    if value in skipPointers:
        return True
    return False
def skip(p1):
    value= skipPointers[skipPointers.index(list1[p1])+1]
    p= list1.index(value)
    return p


def intersectWithSkips(p1,p2):
    global equalComparisons
    answer=[]
    while (p1<len(list1) and p2<len(list2)):
        print('equalComparisons',equalComparisons)
        print('sizecomparison', sizeComparison)
        print('valueP1',list1[p1],'valueP2',list2[p2])


        print('_______________________________________________')
        equalComparisons += 1

        if (list1[p1] == list2[p2]):
            print('match found, ',list1[p1])
            answer.append(list1[p1])
            p1+=1
            p2+=1
            # print('new values are ',list1[p1], list2[p2])
        elif lower(list1[p1],list2[p2]):
            if (hasSkip(list1[p1]) and lowerOrEqual(list1[skip(p1)],list2[p2])):
                while (hasSkip(list1[p1]) and lowerOrEqual(list1[skip(p1)],list2[p2])):
                    print('skipping')
                    p1= skip(p1)
            else:
                p1+=1
        elif (False and False): #list 2 do not have skip pointers

            pass
        else:
            p2+=1
    return answer
answer=intersectWithSkips(0,0)


print('equalComparisons',equalComparisons)
print('sizeComparison', sizeComparison)
print('answer',answer)