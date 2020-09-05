sen = ''
while sen != 'e':
    sen = str(input())
    if sen == 'e':
        break
    sen = sen.split()
    a = float(sen[0])
    b = float(sen[2])
    op = sen[1]
    if op == '+':
        print(a+b)
    if op == '-':
        print(a-b)
    if op == '*':
        print(a*b)
    if op == '/':
        print(a/b)
