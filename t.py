a = "<<>>"
b = "vvvv"
ans = ""
for i in range(0,2):
    ans += a[i]
ans += b
for i in range(2,4):
    ans += a[i]
print(ans)