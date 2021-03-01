import csv

path = "dataset/short_train.csv"

def parse_label(label):
    vec_ind = label.split("|")
    out = [int(str(x) in vec_ind) for x in range(19)]
    return out

def parsing_csv(path):
    with open(path, newline='') as f:
        out = []
        reader = csv.reader(f)
        data = list(reader)
        for row in data[1:]:
            out.append((row[1], parse_label(row[2])))
    return out

parsing_csv(path)



parse_label("3|5")