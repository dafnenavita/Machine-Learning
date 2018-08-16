K-Means Clustering Implementation
---------------------------------

Tech/framework used:

	OS:	Windows 8
	Python 3.6.3 :: Anaconda custom (64-bit)
	Pandas : 0.20.3
--------------------------

Language Used:	Python

-------------------------------------

Packages Used:	nltk, pandas, numpy
Run in: Anaconda Prompt
-------------------------------------

How to compile and run?

Place the tweets file and initial seeds file in the same folder as the code and run the below command

Give the path and arguments as follows:
python tweets-k-means.py 25 InitialSeeds.txt Tweets.json output_file.txt

(or)
python filename <numberOfClusters> <initialSeedsFile> <TweetsDataFile> <outputFile>

-----------------------------------------------------------------------------------------------

Other things to note:

K cannot be greater than the number of seeds in InitialSeeds file, (here 25)

