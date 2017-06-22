import thulac

end = "/SENTENCE_END/"
result = ""

thulac = thulac.thulac(seg_only=True)
line_index = 1

with open("raw.conv") as fd:
    tmp = []
    for line in fd:
        print("line",line_index,sep=" ")
        line_index += 1

        if line.startswith("M"):
            line = line.replace("M ","",1)
            line = line.strip()
            line = [token[0] for token in thulac.cut(line)]
            line = "/".join(line)
            if len(tmp) == 2:
                #print(end.join(tmp) + "\n")
                result += "SENTENCE_START/" + end.join(tmp) + "/SENTENCE_END\n"
                tmp = []
            tmp.append(line)

with open("train.txt","w") as fd:
    fd.write(result)
