import os,re,shutil
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from afinn import Afinn
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.crawler import CrawlerProcess
from  scrapy.exceptions import CloseSpider

# Crawler implementation (Scrapy 2.4.1)
class LinkSpiderSpider(CrawlSpider):
    name = 'link_spider'
    count = 0
    upper_bound = 0
    allowed_domains = ['concordia.ca']
    start_urls = ['https://www.concordia.ca/']
    
    # Makes  sure that crawler obeys robot exclusion standards
    custom_settings = {
        'ROBOTSTXT_OBEY': True
    }
    
    def __init__(self, *args, **kwargs):
        super(LinkSpiderSpider, self).__init__(*args, **kwargs) 
        outputDirectory = os.path.join(os.getcwd(),'resultPages')
        if not os.path.isdir(outputDirectory):
            os.mkdir('resultPages')
        upper_bound = kwargs.get('upper_bound')

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )
    
    def parse(self, response):
        for link in self.link_extractor.extract_links(response):
            yield Request(link.url, callback=self.parse)

    # Checks if number of HTML documents created is less than upper bound value and generates HTML documents
    def parse_item(self, response):
        if self.count < self.upper_bound:
            filename = 'resultPages/'+ 'page'+str(self.count) + '.html'
            print(" Link Extracted : " + response.url+"\n")
            with open(filename, 'w') as f:
                    text = response.body.decode("utf-8") 
                    text = text.encode("ascii","ignore")
                    f.write(text.decode())
                    self.count = self.count+1
        else:
            raise CloseSpider('Bound_exceeded')


#Scrap and Clustering implementation (Require Scikit-learn 0.24.1, BeautifulSoup4 4.10.0)
class ScrapAndCluster:
    
    def __init__(self, upBound):
        self.upBound = upBound

    def scrapeText(self):
        dict = {}
        workingDirectory = os.path.join(os.getcwd(),'resultPages')
        for file in os.listdir(workingDirectory):
            if file.endswith(".html"):
                output = ''
                with open(os.path.join(workingDirectory,file),'r',encoding="utf-8") as fp:
                    contents = fp.read()
                    soup = BeautifulSoup(contents, 'lxml')
                    soup = soup.find('section', id="content-main")
                    if soup is not None:
                        for s in soup(['script', 'style']):
                            s.decompose()
                        output = ' '.join(soup.stripped_strings)
                        output = output.replace('\xa0'," ")

                dict[file] = output
        return dict
    
    def getData(self):
        data=[]
        dict = self.scrapeText()
        for key in dict:
            data.append(dict[key])

        return data
    
    def performClustering(self,clusterSize):
        dataset = self.getData()
        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        X = vectorizer.fit_transform(dataset)
        km = KMeans(
            n_clusters=clusterSize,
            init="k-means++",
            max_iter=100,
            n_init=1
        )

        km.fit(X) # Perform K mean clustering
        return km,vectorizer

    def analysis(self):
        k = int(input("Enter cluster size :"))
        while k >= self.upBound:
            print("Should be less than upper bound for number of downloaded HTML files. Enter new value:")
            k = int(input("Enter cluster size :"))

        km,vectorizer  = self.performClustering(k)
        workingDirectory = os.path.join(os.getcwd(),'resultPages')
        filename = os.path.join(workingDirectory, 'cluster-size-'+str(k) + '.txt')
        afinn = Afinn() # Create sentiment analysis AFINN lexicon object
        
        
        # Find top 50 words in a cluster
        s = 50
        
        sort_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()

        # Print on console and also store on disk in file named cluster-size-*.txt
        with open(filename, 'w') as f:
            for i in range(k):
                print("Top 50 words in Cluster %d:" % i, end="")
                f.write("Top 50 words in cluster "+ str(i) + ":" )
                f.write('\n')
                for ind in sort_centroids[i, :50]:
                    print(" %s" % terms[ind], end="")
                    f.write(terms[ind] +' ,')
                    
                f.write('\n')
                f.write("---------------------------------------------"+'\n')
                print()

        
        print()

        # Find cluster sentiment using top 10 percent of words in each cluster
        clusterSentiment = {}
        vocabLen = int(len(vectorizer.vocabulary_)*0.1) # Calculate 10 percent of  total vocabulary length
        
        # Find top 10 percent words for each cluster and make a single string.
        for i in range(k):
            wordString=""
            for ind in sort_centroids[i, :vocabLen]:
                wordString = wordString + terms[ind] + " "
                
            # Calculate afinn score for the string represent top 10 percent of cluster
            clusterSentiment[i] = afinn.score(wordString)
            print("Cluster "+ str(i)+ " scored sentiment value of "+str(clusterSentiment[i])+'\n')

            
    def run(self):
        while True:
            choice = input("Do you wish perform clustering?(Y/N)")
            if choice == 'Y' or choice == 'y':
                self.analysis()
            else:
                break



process = CrawlerProcess(settings={
    "FEEDS": {
        "items.json": {"format": "json"},
    },
})
upBound = input("Enter upper bound for total number of files to be downloaded: ")
process.crawl(LinkSpiderSpider , upper_bound = int(upBound))
process.start()

sc = ScrapAndCluster(int(upBound))
sc.run()