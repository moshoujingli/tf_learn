import sys
from mnist import MnistModel


def getFromName(name):
    if name == "MinstModel":
        return MnistModel()
    else:
        return None


def main():
    caseName = sys.argv[1]
    modelPath = './model/%s.model' % caseName
    caseIns = getFromName(caseName)
    caseIns.train(modelPath)
    caseIns.test(modelPath)


if __name__ == '__main__':
    main()
