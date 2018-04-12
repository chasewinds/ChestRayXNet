import csv


datalist = "/comvol/nfs/datasets/medicine/NIH-CXR8/data.list"
csvfile = "/comvol/nfs/datasets/medicine/NIH-CXR8/Data_Entry_2017.csv"
diseaseDict = {
    "No Finding": 0,
    "Atelectasis": 1,
    "Cardiomegaly": 2,
    "effusion": 3,
    "Infiltration": 4,
    "Mass": 5,
    "Nodule": 6,
    "Pneumonia": 7,
    "Pneumothorax": 8,
    "Consolidation": 9,
    "Edema": 10,
    "Emphysema": 11,
    "Fibrosis": 12,
    "Pleural_Thickening": 13,
    "Hernia": 14
}
labelList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def processLabels(labels):
    disease = labels.split('|')
    for item in disease:
        rank = diseaseDict[item]
        if rank != 0:
            labelList[rank - 1] = 1
    return


def list2str(labellist):
    return "".join([str(item) for item in labelList])


with open(csvfile) as f:
    f_csv = csv.DictReader(f)
    datafile = open(datalist, 'w')
    total = 0
    for row in f_csv:
        processLabels(row["Finding Labels"])
        datafile.write(row["Image Index"] + " " + list2str(labelList) + "\n")
        print(row["Image Index"], labelList)
        total += 1
        labelList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    datafile.close()
