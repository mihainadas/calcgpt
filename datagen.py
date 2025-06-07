output = 'data/calc.txt'
max = 10


with open(output, "w") as f:
    f.writeln = lambda s: f.write(s+"\n")
    for i in range(max):

        for j in range(max):
            e1 = "{}+{}={}".format(i,j,i+j)
            f.writeln(e1)
            if i>j:
                e2="{}-{}={}".format(i,j,i-j)
                f.writeln(e2)