import urllib.request
from bs4 import BeautifulSoup
import string
import re

def fixSentence(str):
	regex = re.compile('[^a-zA-Z ]')
	str = regex.sub('', str)
	
	str = re.sub(r"(\w)([A-Z])", r"\1 \2", str)
	
	return str

def extractUrl():
	urlinput = input("URL :")

	user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
	headers={'User-Agent':user_agent,} 

	request=urllib.request.Request(urlinput,None,headers)
	response =  urllib.request.urlopen(urlinput)
	html = response.read()


	soup = BeautifulSoup(html,'lxml')
	
	# kill all script and style elements
	for script in soup(["script", "style"]):
		script.extract()    # rip it out
	
	text = soup.get_text(strip = True)
	
	tokens = []
	
	for t in text.split("."):
		if " " in t and len(t) > 10:
			tokens.append(fixSentence(t))
	
	'''
	for tata in tokens:
		print(tata)
	'''
	
	return tokens


	
#extractUrl()

