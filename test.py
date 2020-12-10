product = 1
for i in range(1,100000):
    prob = (i+1)/(i+2)
    product *= prob

print(product)