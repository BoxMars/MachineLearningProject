import os


class Data:
    BASE_DIR = ""
    DATA_DIR = ""
    DATA_LIST=[]
    # IMAGE_DATA=[]
    CSV_FILE=[]

    def __init__(self):
        self.BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            ))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        for i in range(len(os.listdir(self.DATA_DIR))):
            self.DATA_LIST.append(
                os.path.join(
                    self.DATA_DIR,
                    os.listdir(self.DATA_DIR)[i]
                ))
        for i in range(len(self.DATA_LIST)):
            # file=[]
            # for j in range(len(os.listdir(self.DATA_LIST[i]))-1):
            #     file.append(
            #         os.path.join(
            #             self.DATA_LIST[i],
            #             os.listdir(self.DATA_LIST[i])[j]
            #         ))
            # self.IMAGE_DATA.append(file)
            self.CSV_FILE.append(
                os.path.join(self.DATA_LIST[i],os.listdir(self.DATA_LIST[i])[len(os.listdir(self.DATA_LIST[i]))-1]))
