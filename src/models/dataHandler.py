#!/usr/bin/env python3

import pysam as ps
import ftplib
import sys
import shutil
import os
from Bio import AlignIO, Seq
from Bio.Align.Applications import ClustalwCommandline

__authors__ = "Marek Zvara, Marek Hrvol"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz"
__description__ = "MBG"

# Example to download particular data
# ./samtools view -b ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/alignment/HG00096.mapped.ILLUMINA.bwa.GBR.low_coverage.20120522.bam 4:99792835-99851788 > HG00096.bam

class DataHandler(object):
    def __init__(self, shouldFetch, dataPath, genes):
        self.shouldFetch = shouldFetch
        self.dataPath = dataPath
        self.genes = genes

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

    def cleanUpDownloading(self, toDir):
        source = './'

        files = os.listdir(source)
        for f in files:
            if f.endswith("bam.bai"):
                shutil.move(source + f, toDir)

    def createOrClearDirAt(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
            print("Dir - content removed at: " + str(path))

        print("Dir - created at: " + str(path))
        os.makedirs(path)

    def fetchAllDataIfNeededFor(self):
        if not self.shouldFetch:
            print("Fetching data not needed")
            return

        self.createOrClearDirAt(self.dataPath)

        print("Fetching data URLs")
        urls = self.ftpGetURLs(count=2)


        bamBaiPath = self.dataPath + "/" + "bam.bai"
        self.createOrClearDirAt(bamBaiPath)

        for (dirName, url) in urls:
            print("Downloading: " + dirName)
            samfile = ps.AlignmentFile(url, "rb")

            for (index, gene) in enumerate(self.genes):
                geneCounter = 0
                with open(self.dataPath + "/" + gene.name + ".fasta", 'a') as outfile:
                    outfile.write(">"+dirName+"."+gene.name+"\n")
                    for read in samfile.fetch(str(gene.chromosone), gene.locationStart, gene.locationEnd):
                        seq = Seq.Seq(read.seq)
                        if read.is_reverse:
                            seq = seq.reverse_complement()
                        geneCounter += len(seq)
                        outfile.write(str(seq))
                    outfile.write('\n')

                    self.genes[index].length = min(self.genes[index].length, geneCounter)

            self.cleanUpDownloading(toDir=bamBaiPath)
            print("Done: " + dirName)

            samfile.close()

        print("Fetching genes ended")

    def msa(self, clustalw2Path):
        for gene in self.genes:
            geneFile = self.dataPath+"/"+gene.name
            extension = ".fasta"

            genePath = geneFile+extension

            cline = ClustalwCommandline(clustalw2Path, infile=genePath)
            assert os.path.isfile(clustalw2Path), "Clustal W executable missing"

            stdout, stderr = cline()

            align = AlignIO.read(geneFile+".aln", "clustal")
            print(align)
            # with open(geneFile + ".msa", 'a') as outfile:
            #     outfile.write(align.format("fasta"))

