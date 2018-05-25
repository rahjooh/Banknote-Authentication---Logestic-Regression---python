import math
f = open('data_banknote_authentication.txt','r')
ds = []
tet = [ 0.0 , 0.0 , 0.0 , 0.0 , 0.0] ; new_tet = []
for line in f :
    ds.append(('1.0,'+line.replace('\n','')).split(','))

for l in ds:print(l)

def heuristic(instans,tet):
    sum = 0.0
    for i,row  in enumerate(instans):
        sum = sum + float(row)* tet[i]


while(True):
    for t  in range(0,4):
        j = 0.0
        #for i, row2 in enumerate(ds):
        for i in range(0,len(ds)):
            print(i)
            print(t)
            print(ds[i][t])
            print(float(ds[i][t])*tet[t])
            h = sum(float(ds[i][t])*tet[t])
            h = 1/ (1+ math.exp(-h))
            j += (h - ds[i][5])*ds[i][t]
        new_tet.append(tet[t] - j)
    print(tet + '   = > ' + new_tet)
    tet = new_tet
    if tet == new_tet : break



