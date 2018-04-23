#!/usr/bin/env python3

import pysam as ps
import ftplib
import sys
import shutil
import os

__authors__ = "Marek Zvara, Marek Hrvol"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz"
__description__ = "MBG"

# Example to download particular data
# ./samtools view -b ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/alignment/HG00096.mapped.ILLUMINA.bwa.GBR.low_coverage.20120522.bam 4:99792835-99851788 > HG00096.bam

class DataHandler(object):
    def __init__(self, shouldFetch):
        self.shouldFetch = shouldFetch

    def getURLFor(self, folderName):
        baseURL = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data"
        return baseURL + "/" + str(folderName) + "/alignment" + str(
            folderName) + ".mapped.ILLUMINA.bwa.GBR.low_coverage.20120522.bam"

    def ftpGetURLs(self, count=-1):
        urls = []

        baseURL = "ftp.1000genomes.ebi.ac.uk"
        mainDict = "/vol1/ftp/phase3/data"
        f = ftplib.FTP(baseURL)
        f.login()
        f.cwd(mainDict)
        subdirectories = f.nlst()
        counter = -1
        for subDir in subdirectories:
            counter += 1
            try:
                f.cwd(subDir + "/alignment")
                fileNamePrefix = subDir + ".mapped.ILLUMINA.bwa"
                files = filter(lambda file: file.startswith(fileNamePrefix), f.nlst())
                files = list(filter(lambda file: file.endswith(".bam"), files))
                file = files[0]  # TODO: asi by malo hodit vynimku
                urls.append((subDir, "ftp://" + baseURL + f.pwd() + "/" + file))
                if counter == count:
                    break
            except:
                sys.stderr.write("Error in " + subDir + "\n")
                pass
            finally:
                if f.pwd() != mainDict:
                    f.cwd(mainDict)

        return urls

    def cleanUpDownloading(self):
        source = './'
        destination = './data/bam.bai/'

        files = os.listdir(source)
        for f in files:
            if f.endswith("bam.bai"):
                shutil.move(source + f, destination)

    def fetchAllDataIfNeededFor(self, genes):
        if not self.shouldFetch:
            print("Fetching data not needed")
            return

        print("Fetching data URLs")
        urls = self.ftpGetURLs()
        for (dirName, url) in urls:
            print("Downloading: " + dirName)
            samfile = ps.AlignmentFile(url, "rb")

            with open("./data/" + dirName + ".bam", 'w') as outfile:
                for read in samfile.fetch('4', 99792835, 99851788):
                    outfile.write(str(read))

            print("Done: " + dirName)

            samfile.close()

        self.cleanUpDownloading()